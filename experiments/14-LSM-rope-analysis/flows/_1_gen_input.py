""" Helper file to generate a input signal

"""
# %%

from typing import Any

import matplotlib.pyplot as plt
import snntorch as snn
import torch
from torch import nn

n = 400  # signal length (simulation step)

# lr = 0.4  # learning rate
error_scope = 30  # n of steps to average over, in order to compute the error


# %% Gen signal with random artifacts
import sys

sys.path += ['../../../', '../../11-LSTM/']
import matplotlib.pyplot as plt
import src11
import torch

from experimentkit_in.funx import pickle_load, pickle_save_dict
from experimentkit_in.generators.time_series import (add_artifacts_to_signal,
                                                     gen_random_artifacts,
                                                     gen_simple_sin,
                                                     ts_gen_signal_shifts)

signal_length = n
sampling_rate = 50

# Expt assumption:
#   alternated categ_1 and categ_2, same n of signals

main_signal = torch.zeros(signal_length * sampling_rate)
num_signals = 29

categ_1_max_length = 5
categ_1_signals = gen_random_artifacts(
    num_signals = num_signals,
    min_length = categ_1_max_length,
    max_length = categ_1_max_length,
    min_frequency = 5,
    max_frequency = 5,
    amplitude=1,
    sampling_rate=sampling_rate)
"""List(np.array): List of signals"""

categ_2_max_length = 5
categ_2_signals = gen_random_artifacts(
    num_signals = num_signals,
    min_length = categ_2_max_length,
    max_length = categ_2_max_length,
    min_frequency = 10,
    max_frequency = 10,
    amplitude=1,
    sampling_rate=sampling_rate)
"""List(np.array): List of signals"""

max_idx = signal_length - max(categ_1_max_length, categ_2_max_length) - 1

categ_1_idxs = []
categ_2_idxs = []
dirty_signal = main_signal.clone()
labels = torch.zeros_like(dirty_signal)
min_spc = 25  # minimum intra-signals space
pp = 0  # pointer
for i in range(num_signals):
    if i > 0:
        pp += min_spc
    cur_sig_1 = categ_1_signals[i]
    cur_idx_start = torch.randint(pp, pp + 100, size=(1, 1)).item()
    cur_idx_end = cur_idx_start + len(cur_sig_1)
    
    if len(dirty_signal) < cur_idx_end:
        print(f"Signal overflow at {i}")
        break
    dirty_signal[cur_idx_start:cur_idx_end] += cur_sig_1
    pp = cur_idx_end
    categ_1_idxs.append((cur_idx_start, cur_idx_end))
    labels[cur_idx_start:cur_idx_end] = 1

    pp += min_spc
    cur_sig_2 = categ_2_signals[i] 
    cur_idx_start = torch.randint(pp, pp + 100, size=(1, 1)).item()
    cur_idx_end = cur_idx_start + len(cur_sig_2)

    if len(dirty_signal) < cur_idx_end:
        print(f"Signal overflow at {i}")
        break
    dirty_signal[cur_idx_start:cur_idx_end] += cur_sig_2
    pp = cur_idx_end
    categ_2_idxs.append((cur_idx_start, cur_idx_end))
    labels[cur_idx_start:cur_idx_end] = 2


# %%
# visualize data set

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
labels_1 = torch.argwhere(labels == 1)
labels_2 = torch.argwhere(labels == 2)
axs[0].plot(main_signal)
axs[0].set_title("Baseline")
axs[1].plot(dirty_signal, label='dirty signal')
for i in range(num_signals):
    idx_from = categ_1_idxs[i][0]
    idx_to = categ_1_idxs[i][1]
    axs[1].plot(
        torch.arange(idx_from, idx_to),
        dirty_signal[idx_from:idx_to], c='k')
for i in range(num_signals):
    idx_from = categ_2_idxs[i][0]
    idx_to = categ_2_idxs[i][1]
    axs[1].plot(
        torch.arange(idx_from, idx_to),
        dirty_signal[idx_from:idx_to], c='orange')
axs[1].set_title("Input signal, $I$")
axs[1].grid()

axs[2].plot(
    labels_1, torch.ones_like(labels_1),
    marker='|', markersize=22, linestyle="", c='k')
axs[2].plot(
    labels_2, torch.ones_like(labels_2),
    marker='|', markersize=22, linestyle="", c='orange')
axs[2].set_ylim(0.9)
axs[2].set_title('labels indices')
axs[2].get_yaxis().set_visible(False)
axs[2].set_xlim(0, 500)
fig.tight_layout()


I = dirty_signal
I = I.clip(min=0)

# # Labels (y) definition
y = labels

fig, axs = plt.subplots(2, 1)
axs[0].plot(torch.arange(len(I)), I)
axs[0].set_title('I')
axs[0].set_ylabel('amplitude')
axs[0].set_xlabel('t')
axs[0].grid()

axs[1].plot(torch.arange(len(y)), y, 'o')
axs[1].set_title('Labels (y)')
axs[1].set_ylabel('')
axs[1].set_xlabel('t')
axs[1].set_yticks([-1, 0, 1])
axs[1].set_ylim(0, 2.3)
axs[1].grid()
fig.tight_layout()
# %%