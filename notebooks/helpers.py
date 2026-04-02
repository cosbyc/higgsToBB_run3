import awkward as ak
import numpy as np
import operator
bound_ops = {'min': operator.ge, 'max': operator.le}

import matplotlib.pyplot as plt
import mplhep as hep

import hist
from itertools import product


# class BDTFeatureExtractor(processor.ProcessorABC):
#     def __init__(self, signal_label="signal", background_label="background"):
#         self.signal_label = signal_label
#         self.background_label = background_label


#     def process(self, events):
#         dataset = events.metadata.get('short_name', 'unknown')
#         is_signal = events.metadata.get('isSignal', False)
        
#         jets = events.Jet[events.Jet.pt > 20]        
#         b_jets = jets[jets.btagPNetB > 0.03]
        
#         pairs = ak.combinations(b_jets, 2, fields=["b1", "b2"])
        
#         dijet = pairs.b1 + pairs.b2
#         dijet_pt = dijet.pt
#         dijet_eta = dijet.eta
#         dijet_phi = dijet.phi
#         dijet_mass = dijet.mass
        
#         jet_1 = pairs.b1;                    jet_2 = pairs.b2
#         jet1_pt = jet_1.pt;                  jet2_pt = jet_2.pt
#         jet1_eta = jet_1.eta;                jet2_eta = jet_2.eta    
#         jet1_phi = jet_1.phi;                jet2_phi = jet_2.phi
#         jet1_btag = jet_1.btagPNetB;         jet2_btag = jet_2.btagPNetB
#         if is_signal:
#             jet1_flav = jet_1.hadronFlavour;  jet2_flav = jet_2.hadronFlavour
#             bbarmask = (((jet1_flav - jet2_flav) == 0) & (abs(jet1_flav) == 5))
        
#         delta_phi = jet_1.delta_phi(jet_2)
#         delta_r = jet_1.delta_r(jet_2)
        

#         features = {
#             'dijet_pt': ak.flatten(dijet_pt), 'dijet_eta': ak.flatten(dijet_eta), 'dijet_phi': ak.flatten(dijet_phi), 'dijet_mass': ak.flatten(dijet_mass),
#             'jet1_pt': ak.flatten(jet1_pt), 'jet2_pt': ak.flatten(jet2_pt),
#             'jet1_eta': ak.flatten(jet1_eta), 'jet2_eta': ak.flatten(jet2_eta),
#             'jet1_phi':     ak.flatten(jet1_phi), 'jet2_phi': ak.flatten(jet2_phi),
#             'jet1_btag': ak.flatten(jet1_btag), 'jet2_btag': ak.flatten(jet2_btag),
#             'delta_phi': ak.flatten(delta_phi), 'delta_r': ak.flatten(delta_r),
#         } 

#         dijet_array = ak.flatten(dijet_pt)
#         total_pairs = ak.num(dijet_array, axis=0)
#         n_events = ak.num(events, axis=0)

#         if is_signal:
#             labels_flat = ak.values_astype(ak.flatten(bbarmask), np.int32)
#         else:        
#             labels_flat = ak.zeros_like(dijet_array)
        
#         return {
#             'features': {k: column_accumulator(v) for k, v in features.items()},
#             'labels': column_accumulator(labels_flat),
#             'n_events': n_events,
#             'n_pairs': total_pairs,
#         }
    
#     def postprocess(self, accumulator):
#         pass

def correctAndGENMatch(events, is_gen=False):
    muons = events.Muon
    jets = events.Jet

    parkingMuonMask = (muons.pt > 11) & (abs(muons.eta) < 0.9)
    muons["isTrigMuon"] = muons.triggerIdLoose & parkingMuonMask
    jets["hasTrigMuon"] = ~ak.is_none(jets.nearest(muons[muons.isTrigMuon], threshold=0.4), axis=1)

    corrected_jets = jets
    corrected_jets["pt"] = corrected_jets.pt*(1-corrected_jets.rawFactor)*corrected_jets.UParTAK4RegPtRawCorrNeutrino
    events["CorrJet"] = corrected_jets

    if is_gen:
        events["GenPart"] = gen_higgs_parentage(events)
        gens = events.GenPart

        gen_higgs = gens[gens.pdgId == 25]
        higgs = gen_higgs[gen_higgs.hasFlags(['isLastCopy'])]
        events["GenHiggs"] = higgs

        gen_muons = gens[abs(gens.pdgId) == 13]
        gen_muons = gen_muons[gen_muons.hasFlags(['isLastCopy'])]
        events["GenMuon"] = gen_muons
        muons["higgsGenMatched"] = ~ak.is_none(muons.nearest(events.GenMuon[events.GenMuon.fromHiggs], threshold=0.1), axis=1)
        events["Muon"] = muons

        bgens = events.GenPart[(abs(events.GenPart.pdgId) == 5)]
        events["GenBPart"] = bgens

        corrected_jets["bGenMatched"] = abs(corrected_jets.partonFlavour) == 5
        corrected_jets["higgsGenMatched"] =  ~ak.is_none(corrected_jets.nearest(events.GenBPart[events.GenBPart.fromHiggs], threshold=0.4), axis=1)
        events["CorrJet"] = corrected_jets

    return events
    

def get_latex_label(var_name):
    if var_name in var_to_latex:
        return var_to_latex[var_name]
    return var_name

def gen_higgs_parentage(events):
    # is_higgs = abs(events.GenPart.pdgId) == 25
    is_higgs = abs(events.GenPart.pdgId) == 25
    current_mask = is_higgs
    mother_indices = events.GenPart.genPartIdxMother
    
    while ak.any(mother_indices >= 0):
        has_mother = mother_indices >= 0
        mother_is_higgs = ak.fill_none(ak.mask(is_higgs[mother_indices], has_mother), False)

        current_mask = current_mask | mother_is_higgs
        mother_indices = ak.fill_none(ak.mask(events.GenPart.genPartIdxMother[mother_indices], has_mother), -1 )
    
    GenPart = ak.with_field(events.GenPart, current_mask, where="fromHiggs")
    return GenPart


def get_yields(signal_events, background_events, cuts_dict=None, mass_window=[100, 150]):
    def parse_cut(cut_key, threshold):
        level, var_name, bound = cut_key.split('_')
        op = bound_ops[bound]
        if level == 'j':  # jet cut
            return lambda jets: op(getattr(jets, var_name), threshold)
        else:  # higgs-cand. cut
            if var_name == 'deltaR': return lambda pairs: op(pairs['0'].deltaR(pairs['1']), threshold)
            elif var_name == 'PAIReDbb': 
                def get_bb_score_for_pairs(pairs, events, pair_jet1_idx, pair_jet2_idx):
                    return get_PAIReD_bb_scores_for_pairs(events, pair_jet1_idx, pair_jet2_idx)
                
                return lambda pairs, events, pair_jet1_idx, pair_jet2_idx: op(get_bb_score_for_pairs(pairs, events, pair_jet1_idx, pair_jet2_idx), threshold)
            else: return lambda pairs: op(getattr(pairs['0'] + pairs['1'], var_name), threshold)
    
    def apply_cuts(events, is_signal=True):
        jet_mask = ak.ones_like(events.Jet.pt, dtype=bool)
        if cuts_dict is not None:
            for cut_key, threshold in cuts_dict.items():
                if cut_key.startswith('j_'):
                    cut_func = parse_cut(cut_key, threshold)
                    temp_mask = cut_func(events.Jet)
                    jet_mask = jet_mask & temp_mask
        passed_jets = events.Jet[jet_mask]
        

        all_jet_indices = ak.local_index(events.Jet)
        all_pairs = ak.argcombinations(events.Jet, 2)
        pair_jet1_mask = jet_mask[all_pairs['0']]
        pair_jet2_mask = jet_mask[all_pairs['1']]

        pair_mask = pair_jet1_mask & pair_jet2_mask
        

        pair_jet1_idx = all_pairs['0'][pair_mask]
        pair_jet2_idx = all_pairs['1'][pair_mask]

        
        # Create the actual jet pairs from passed_jets
        pairs = ak.combinations(passed_jets, 2)
        pair_mass = (pairs['0'] + pairs['1']).mass
        
        mass_mask = (pair_mass >= mass_window[0]) & (pair_mass <= mass_window[1])

        if cuts_dict is not None:
            for cut_key, threshold in cuts_dict.items():
                if cut_key.startswith('p_'):
                    cut_func = parse_cut(cut_key, threshold)
                    if cut_key == 'p_PAIReDbb_min' or cut_key == 'p_PAIReDbb_max':
                        temp_mask = cut_func(pairs, events, pair_jet1_idx, pair_jet2_idx)
                    else:
                        temp_mask = cut_func(pairs)
                    mass_mask = mass_mask & temp_mask

        pairs = pairs[mass_mask]
        pair_mass = pair_mass[mass_mask]
        
        has_valid_pair = ak.any(ak.ones_like(pair_mass), axis=1)
        return float(ak.sum(has_valid_pair)), pair_mass
    
    signal_yield, signal_mass = apply_cuts(signal_events, is_signal=True)
    background_yield, background_mass = apply_cuts(background_events, is_signal=False)
    return signal_yield, background_yield, signal_mass, background_mass


def scan_parameters(signal_events, background_events, scan_params, flat_cuts={}, mass_window=[100, 150]):
    param_names = list(scan_params.keys())
    param_values = [scan_params[name] for name in param_names]

    shape = tuple(len(values) for values in param_values)
    
    significances = np.zeros(shape)
    signal_yields = np.zeros(shape)
    background_yields = np.zeros(shape)
    # optimal_windows = np.zeros(shape + (2,))
    
    value_combinations = list(product(*param_values))
    
    for idx, values in enumerate(value_combinations):
        n_idx = np.unravel_index(idx, shape)
        
        current_cuts = flat_cuts.copy()
        for name, value in zip(param_names, values):
            current_cuts[name] = value
        signal_yield, background_yield, signal_mass, background_mass = get_yields(
            signal_events, background_events, current_cuts, mass_window
        )
        
        # opt_window, max_sig = find_optimal_mass_window(signal_mass, background_mass)
        # optimal_windows[n_idx] = opt_window
        signal_yields[n_idx] = signal_yield
        background_yields[n_idx] = background_yield
        significances[n_idx] = min(signal_yield / np.sqrt(background_yield), signal_yield)

    # Find best point
    best_idx = np.unravel_index(np.argmax(significances), significances.shape)
    best_cuts = flat_cuts.copy()
    for name, values, i in zip(param_names, param_values, best_idx):
        best_cuts[name] = values[i]

    signal_yield, background_yield, signal_mass, background_mass = get_yields(
        signal_events, background_events, best_cuts, mass_window
    )

    return {
        'parameters': param_names,
        'values': param_values,
        'significances': significances,
        'signal_yields': signal_yields,
        'background_yields': background_yields,
        # 'optimal_windows': optimal_windows,
        'best_cuts': best_cuts,
        # 'best_window': best_window,
        'best_significance': significances[best_idx],
        'best_signal_mass': signal_mass,
        'best_background_mass': background_mass
    }


# def find_optimal_mass_window(signal_mass, background_mass, min_mass=60, max_mass=160, step=5):
#     best_significance = 0
#     best_window = (min_mass, max_mass)
    
#     signal_mass = ak.flatten(signal_mass)
#     background_mass = ak.flatten(background_mass)
    
#     for window_size in range(10, int((max_mass - min_mass)/step), 2):
#         for start in np.arange(min_mass, max_mass - window_size*step, step):
#             end = start + window_size*step
#             signal_count = np.sum((signal_mass >= start) & (signal_mass <= end))
#             background_count = np.sum((background_mass >= start) & (background_mass <= end))
            
#             significance = min(signal_count / np.sqrt(background_count), signal_count)
#             if significance > best_significance:
#                 best_significance = significance
#                 best_window = (start, end)
    
#     return best_window, best_significance

def get_PAIReD_bb_scores_for_pairs(events, pair_jet1_idx, pair_jet2_idx):

    paired_idx1 = events.PAIReDJets.idx_jet1
    paired_idx2 = events.PAIReDJets.idx_jet2
    paired_bb_scores = events.PAIReDJets.bb_score
    
    #  match mask
    match_mask = (pair_jet1_idx[:, :, None] == paired_idx1[:, None, :]) & (pair_jet2_idx[:, :, None] == paired_idx2[:, None, :])
    
    # check for matches
    match_indices = ak.argmax(match_mask, axis=-1)
    has_match = ak.any(match_mask, axis=-1)
    
    #  if match, get score, otherwise 0
    matched_scores = paired_bb_scores[match_indices]
    bb_scores = ak.where(has_match, matched_scores, 0.0)
    
    return bb_scores




def slice_4d_histogram(hist_4d, cuts=None, project_axes=None, return_numpy=True):
    """
    Slice a 4D histogram with physical values and return numpy histograms for plotting.
    
    Parameters:
    -----------
    hist_4d : hist.Hist
        The 4D histogram from the dijet analysis
    cuts : dict, optional
        Dictionary of cuts in physical units. Keys are axis names, values are tuples (min, max).
        Example: {'dijet_mass': (100, 200), 'lower_btag': (0.6, 1.0)}
    project_axes : list, optional
        List of axes to project to (1D or 2D). If None, returns the full sliced histogram.
        Example: ['dijet_mass'] for 1D, ['dijet_mass', 'dijet_pt'] for 2D
    return_numpy : bool, default True
        If True, returns numpy histogram tuples (counts, bins) for 1D or (counts, x_bins, y_bins) for 2D.
        If False, returns the hist object.
    
    Returns:
    --------
    If return_numpy=True:
        - For 1D: (counts, bins) tuple like numpy.histogram()
        - For 2D: (counts, x_bins, y_bins) tuple
        - For higher dim: hist object
    If return_numpy=False:
        - hist object
    
    Example:
    --------
    # Get dijet mass distribution with b-tag > 0.6
    counts, bins = slice_4d_histogram(
        hist_4d, 
        cuts={'lower_btag': (0.6, 1.0)}, 
        project_axes=['dijet_mass']
    )
    # Use: counts, bins (standard numpy histogram format)
    
    # Get 2D plot: dijet_pt vs dijet_mass with jet pt > 50 GeV
    counts, x_bins, y_bins = slice_4d_histogram(
        hist_4d,
        cuts={'subleading_pt': (50, 1000)},
        project_axes=['dijet_mass', 'dijet_pt']
    )
    # Use: counts, x_bins, y_bins
    """
    
    # Get axis information
    axis_names = [axis.name for axis in hist_4d.axes]
    axis_edges = [axis.edges for axis in hist_4d.axes]
    
    # Convert physical values to bin indices
    slice_dict = {}
    if cuts:
        for axis_name, (min_val, max_val) in cuts.items():
            if axis_name not in axis_names:
                raise ValueError(f"Axis '{axis_name}' not found. Available axes: {axis_names}")
            
            axis_idx = axis_names.index(axis_name)
            edges = axis_edges[axis_idx]
            
            # Find bin indices for the range
            min_bin = np.searchsorted(edges, min_val, side='right') - 1
            max_bin = np.searchsorted(edges, max_val, side='left')
            
            # Ensure valid range
            min_bin = max(0, min_bin)
            max_bin = min(len(edges) - 1, max_bin)
            
            if min_bin >= max_bin:
                print(f"Warning: No bins found for {axis_name} in range [{min_val}, {max_val}]")
                min_bin = 0
                max_bin = 1
            
            slice_dict[axis_name] = slice(min_bin, max_bin)
            print(f"Applied cut {axis_name}: [{min_val}, {max_val}] -> bins {min_bin}:{max_bin}")
    
    # Apply cuts
    if slice_dict:
        sliced_hist = hist_4d[slice_dict]
    else:
        sliced_hist = hist_4d
    
    # Project to specified axes
    if project_axes:
        if len(project_axes) == 1:
            # 1D projection
            projected_hist = sliced_hist.project(project_axes[0])
            if return_numpy:
                counts = projected_hist.values()
                bins = projected_hist.axes[0].edges
                return counts, bins
            else:
                return projected_hist
                
        elif len(project_axes) == 2:
            # 2D projection
            projected_hist = sliced_hist.project(project_axes[0], project_axes[1])
            if return_numpy:
                x_bins = projected_hist.axes[0].edges
                y_bins = projected_hist.axes[1].edges
                counts_2d = projected_hist.values()
                return counts_2d, x_bins, y_bins
            else:
                return projected_hist
        else:
            # Higher dimensional - return hist object
            projected_hist = sliced_hist.project(*project_axes)
            return projected_hist
    else:
        # Return the sliced histogram as-is
        return sliced_hist


def plot_histogram_slice(hist_4d, cuts=None, project_axes=None, title=None, **plot_kwargs):
    """
    Convenience function to slice and plot a histogram in one step.
    
    Parameters:
    -----------
    hist_4d : hist.Hist
        The 4D histogram from the dijet analysis
    cuts : dict, optional
        Dictionary of cuts in physical units
    project_axes : list
        List of axes to project to (1D or 2D)
    title : str, optional
        Plot title
    **plot_kwargs
        Additional arguments passed to matplotlib plotting functions
    
    Returns:
    --------
    matplotlib figure and axis objects
    """
    import matplotlib.pyplot as plt
    
    if len(project_axes) == 1:
        # 1D plot
        counts, bins = slice_4d_histogram(hist_4d, cuts, project_axes, return_numpy=True)
        centers = (bins[:-1] + bins[1:]) / 2  # Calculate bin centers
        
        fig, ax = plt.subplots(figsize=plot_kwargs.get('figsize', (8, 6)))
        ax.step(centers, counts, where='mid', linewidth=2, **plot_kwargs)
        ax.fill_between(centers, counts, step='mid', alpha=0.3, **plot_kwargs)
        ax.set_xlabel(project_axes[0].replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Number of Events', fontsize=12)
        ax.set_title(title or f"{project_axes[0].replace('_', ' ').title()} Distribution", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
    elif len(project_axes) == 2:
        # 2D plot
        counts, x_bins, y_bins = slice_4d_histogram(hist_4d, cuts, project_axes, return_numpy=True)
        x_centers = (x_bins[:-1] + x_bins[1:]) / 2  # Calculate bin centers
        y_centers = (y_bins[:-1] + y_bins[1:]) / 2
        
        fig, ax = plt.subplots(figsize=plot_kwargs.get('figsize', (10, 8)))
        im = ax.pcolormesh(x_centers, y_centers, counts.T, **plot_kwargs)
        ax.set_xlabel(project_axes[0].replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(project_axes[1].replace('_', ' ').title(), fontsize=12)
        ax.set_title(title or f"{project_axes[0].replace('_', ' ').title()} vs {project_axes[1].replace('_', ' ').title()}", fontsize=14)
        plt.colorbar(im, ax=ax, label='Number of Events')
        
    else:
        raise ValueError("project_axes must have 1 or 2 elements for plotting")
    
    plt.tight_layout()
    return fig, ax