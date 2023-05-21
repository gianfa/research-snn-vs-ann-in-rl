""" Utility functions 

"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time

import torch

def compare_sample_from_data(
    data: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    n_samples: int = 3
) -> np.ndarray:
    sampled_idxs = torch.randint(len(data), (1, n_samples)).flatten()

    fig, axs = plt.subplots(len(sampled_idxs), 3, figsize=(4, 6))
    for i, idx in enumerate(sampled_idxs):
        img = data[idx]
        axs[i, 0].imshow(img.squeeze())
        axs[i, 1].text(
            0.5, 0.5, str(y_true[idx].item()), horizontalalignment='center',
        verticalalignment='center', fontsize=25)
        axs[i, 1].axis('off')
        axs[i, 2].text(
            0.5, 0.5, str(y_pred[idx].item()), horizontalalignment='center',
        verticalalignment='center', fontsize=25)
        axs[i, 2].axis('off')
    return axs


def pickle_save_dict(fpath: str, d: dict) -> str:
    with open(fpath, "wb") as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not os.path.exists(fpath):
        raise Exception(f"File '{fpath}' saving failed!")
    return fpath


def pickle_load(fpath: str):
    if os.path.getsize(fpath) == 0:
        print(f"The file '{fpath}' is empty")
    with open(fpath, "rb") as handle:
        f = pickle.load(handle)
    return f


def generate_random_name(length: str = 22, with_timestamp: bool = True) -> str:
    bank = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    tstamp = str(time.time()).replace(".", "d") if with_timestamp else ""
    to_len = length - len(tstamp)
    if to_len < 0:
        raise Exception(
            "If you want generate a random name with time_stamp, " +
            f"length must be greater than 18")
    rand_chars = "".join(
        [bank[ci] for ci in np.random.choice(len(bank), to_len)])
    return f"{tstamp}-{rand_chars}"


def moving_average(signal: np.array, window_size: int = None) -> np.array:
    if not window_size:
        window_size = max(round(len(signal) * 0.05), 1)
    window = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, window, mode='same')
    return smoothed_signal