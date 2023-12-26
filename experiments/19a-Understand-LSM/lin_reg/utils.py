""" Helper file to generate a input signal

"""
# %%

from pathlib import Path
import sys
sys.path += ['../../../']
from typing import Any, Tuple

import matplotlib.pyplot as plt
import torch

import experimentkit_in as ek
from experimentkit_in.generators.time_series import gen_random_artifacts


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


# %% Test

# (main_signal, labels_bounds) = ts_gen_signal_with_sin_fragments()

# plt.plot(range(len(main_signal)), main_signal)
# for lbi in labels_bounds:
#     lbi_from, lbi_to = lbi
#     plt.plot(
#         range(lbi_from, lbi_to),
#         main_signal[lbi_from:lbi_to],
#         color='orange')

