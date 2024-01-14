""" Helper file to generate a input signal

"""
# %%

from pathlib import Path
import sys
sys.path += ['../../../']
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import torch

import experimentkit_in as ek
from experimentkit_in.generators.time_series import gen_random_artifacts

logger = ek.logger_config.setup_logger(__file__)


def ts_gen_signal_with_sin_fragments(
    signal_length = 400,
    sampling_rate = 50,
    baseline = 0,
    fragment_min_length = 5,
    fragment_max_length = 5,
    sig_coverage = 0.2,
    fpath = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gen signal with random artifacts """

    if fpath is not None:
        data = ek.funx.pickle_load(fpath)
        return data['signal'], data['labels_bounds']

    max_num_signals = signal_length // fragment_min_length

    main_signal = torch.zeros(signal_length * sampling_rate) + baseline

    categ_1_signals = gen_random_artifacts(
        num_signals = max_num_signals,
        min_length = fragment_min_length,
        max_length = fragment_max_length,
        min_frequency = 5,
        max_frequency = 5,
        amplitude=1,
        sampling_rate=sampling_rate)
    """List(np.array): List of signals"""

    num_signals = int(max_num_signals * sig_coverage)

    # draw candidate positions where to add the fragments
    pos_candidates = torch.randperm(
        signal_length * sampling_rate)[: num_signals].tolist()

    labels_bounds = []
    for i, pci in enumerate(pos_candidates):
        fragment = categ_1_signals[i]
        try:
            main_signal[pci: pci + len(fragment)] = \
                torch.tensor(fragment).flatten()
            labels_bounds.append((pci, pci + len(fragment)))
        except:
            print(i, len(fragment))
    
    return (main_signal, labels_bounds)


def ts_gen_signal_with_multisin_fragments(
    categs: List[dict] = [
        {
            'freq': (10, 10),
            'length': (5, 10),
            'amplitude': 1,
            'coverage': 0.4,
        },
        {
            'freq': (20, 20),
            'length': (5, 10),
            'amplitude': 1,
            'coverage': 0.4,
        }],
        signal_length: int = 400,
        sampling_rate: int = 50,
        baseline: float = 0,
        sig_coverage: float = 0.2,
        fpath: str = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gen signal with random artifacts
    
    Example
    -------
    >>> (main_signal, labels_bounds) = ts_gen_signal_with_multisin_fragments(
    ...     categs,
    ...     signal_length = 400,
    ...     sampling_rate = 50,
    ...     baseline = 0,
    ...     sig_coverage = 0.55,
    ...     fpath = None,
    ... )
    """

    if fpath is not None:
        data = ek.funx.pickle_load(fpath)
        return data['signal'], data['labels_bounds']

    _categs = categs.copy()

    # Create the base signal
    main_signal = torch.zeros(signal_length * sampling_rate) + baseline

    # Collect the many fragments for each category
    categ_fragments = []
    for ci in _categs:
        if len(ci['freq']) == 2:
            freq_min = ci['freq'][0]
            freq_max = ci['freq'][1]
        elif len(ci['freq']) == 1:
            freq_min = ci['freq']
            freq_max = ci['freq']
        if len(ci['length']) == 2:
            length_min = ci['length'][0]
            length_max = ci['length'][1]
        elif len(ci['length']) == 1:
            length_min = ci['length']
            length_max = ci['length']

        max_num_signals = signal_length // length_min

        categ_fragments.append(
            gen_random_artifacts(
                num_signals=max_num_signals,
                min_length=length_min,
                max_length=length_max,
                min_frequency=freq_min,
                max_frequency=freq_max,
                amplitude=ci['amplitude'],
                sampling_rate=sampling_rate)
        )

        ci['max_num'] = max_num_signals


    # compute the num of signals requested by the coverages
    coverages = torch.tensor([ci['coverage'] for ci in _categs])
    max_nums = torch.tensor([ci['max_num'] for ci in _categs])
    coverages =  (coverages / coverages.sum())
    num_signals = (max_nums * coverages * sig_coverage).to(int)


    # categs distribution
    # random choice categs
    choices = torch.multinomial(
        torch.tensor(num_signals).to(float),
        num_signals.sum(),
        replacement=True)

    max_inner_spc = 700  # max_space_between_fragments
    labels_bounds = [[] for ci in range(len(_categs))]
    pos_start = 0
    pos_end = 0
    for i, fragment_id in enumerate(choices):
        fragment = categ_fragments[fragment_id][i]

        pos_start = pos_end + torch.randint(0, max_inner_spc, size=(1, 1)).item()
        pos_end = pos_start + len(fragment)

        try:
            main_signal[pos_start: pos_end] = \
                torch.tensor(fragment).flatten()
            labels_bounds[fragment_id].append((pos_start, pos_end))
        except Exception as e:
            logger.warning(f"{i}; len(fragment): {len(fragment)}; e:{e}")
        
    return (main_signal, labels_bounds)


def plot_signal_categs(signal, labels_bounds, ax = None):
    """
    
    Example
    -------
    >>> (main_signal, labels_bounds) = ts_gen_signal_with_multisin_fragments()
    >>> plot_signal_categs(main_signal, labels_bounds)
    """
    if ax is None:
        _, ax = plt.subplots()

    colormap = plt.cm.get_cmap('tab10', len(labels_bounds))
    colors = [colormap(i) for i in range(len(labels_bounds))]

    ax.plot(signal, color='k')
    for ci, lb_i in enumerate(labels_bounds):
        for xj_0, xj_1 in lb_i:
            try:
                xi = torch.arange(xj_0, xj_1)
                yi = signal[xj_0: xj_1]
                ax.plot(xi, yi, color=colors[ci])
            except Exception as e:
                logger.warning(f"WARN| 0,0; {ci}: {e}")
    return ax


def split_tensor_by_integer_groups(tensor):
    """
    Example
    -------
    >>> tns = torch.tensor([0,0,0,3,1,1,2,2,2,2,1,0])
    >>> res = split_tensor_by_integer_groups(tns)
    """
    result = {}
    start_idx = 0
    current_val = tensor[0]

    for i in range(1, len(tensor)):
        if tensor[i] != current_val:
            # add the current sequence to the result
            if current_val.item() not in result:
                result[current_val.item()] = []
            result[current_val.item()].append([start_idx, i - 1])
            
            # update the current value and the start index
            current_val = tensor[i]
            start_idx = i

    # add last seq
    if current_val.item() not in result:
        result[current_val.item()] = []
    result[current_val.item()].append([start_idx, len(tensor) - 1])

    # convert the result to a ordered list
    ordered_result = []
    for key in sorted(result):
        ordered_result.append(result[key])

    return ordered_result