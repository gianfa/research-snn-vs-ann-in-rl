""" Helper file to generate a input signal

"""
# %%

from typing import Any

import matplotlib.pyplot as plt
import snntorch as snn
import torch
from torch import nn

import os
import pickle
from pathlib import Path
import sys

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

sys.path += ['../../']

from _0_config import *

CROSSBRAIN_DIR = (ROOT/'../CROSSBRAIN_signals').resolve()  # CROSSBRAIN_signals dir
DISK_DIR = Path('/DATA1/crossbrain_data/')
DATA_DIR = CROSSBRAIN_DIR/'data'

# %%

assert CROSSBRAIN_DIR.resolve().name == "CROSSBRAIN_signals"
assert DISK_DIR.exists(), f"DISK_DIR id not a valid path"


# %%

dataset_path: str or Path = DISK_DIR/"ieeg-sweez-eth_3patients/dataset.p"
# dataset_path: str or Path = DISK_DIR/"ieeg-sweez-eth_3patients/dataset_BSA_encoded_bool.p"
dataset_is_spike_encoded = False

ds = pickle.load(open(dataset_path, "rb"))

fragment_labels = torch.tensor([ds[i][0] for i in range(len(ds)) ])
fragment_signals = torch.tensor(np.array([ds[i][1] for i in range(len(ds)) ]))  # voltage signal

print(
    f"fragment_labels.shape: {fragment_labels.shape}\fragment_signals.shape: {fragment_signals.shape}\n"
)

# %%
"""
Shuffle the dataset in order to have a random collection of labeled signals

"""

fig, axs = plt.subplots(1, 2)
axs[0].plot(fragment_labels)
axs[0].set_title("Labels, from loading")

new_pos = torch.randperm(len(fragment_labels))
axs[1].plot(fragment_labels[new_pos])
axs[1].set_title("Labels, after permutation")
fig.tight_layout

# %%

fragment_signals = fragment_signals[new_pos]
fragment_labels = fragment_labels[new_pos]

# %%

if dataset_is_spike_encoded:
    I = fragment_signals

    # # Labels (y) definition
    y = fragment_labels

else:
    num_signals = 30
    signal_length = fragment_signals.shape[1] * (num_signals + 10) # signal length (simulation step)

    dirty_signal = torch.zeros(signal_length)

    min_spc = 25  # minimum intra-signals space
    pp = 0  # pointer
    sig_idxs = []
    sig_labels = []
    for i in range(num_signals):
        if i > 0:
            pp += min_spc
        cur_sig = fragment_signals[i]
        cur_idx_start = torch.randint(pp, pp + 100, size=(1, 1)).item()
        cur_idx_end = cur_idx_start + len(cur_sig)
        
        if len(dirty_signal) < cur_idx_end:
            print(f"Signal overflow at {i}")
            break
        dirty_signal[cur_idx_start:cur_idx_end] += cur_sig
        pp = cur_idx_end
        sig_idxs.append((cur_idx_start, cur_idx_end))
        sig_labels[cur_idx_start:cur_idx_end] = torch.ones(cur_idx_end-cur_idx_start) * fragment_labels[i]

    sig_idxs = torch.tensor(sig_idxs)
    sig_labels = torch.tensor(sig_labels)


    # visualize data set

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 10))

    axs[0].plot(dirty_signal, label='dirty signal')
    for i in range(num_signals):
        idx_from = sig_idxs[i][0]
        idx_to = sig_idxs[i][1]
        axs[0].plot(
            torch.arange(idx_from, idx_to),
            dirty_signal[idx_from:idx_to], c='k')
    axs[0].set_title("Input signal, $I$")
    axs[0].grid()

    axs[1].plot(
        sig_labels, torch.ones_like(sig_labels),
        marker='|', markersize=22, linestyle="", c='k')
    axs[1].set_ylim(0.9)
    axs[1].set_title('labels indices')
    axs[1].get_yaxis().set_visible(False)
    axs[1].set_xlim(0, 500)
    fig.tight_layout()


    I = dirty_signal

    # # Labels (y) definition
    y = sig_labels

fig, axs = plt.subplots(2, 1, sharex=True)
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