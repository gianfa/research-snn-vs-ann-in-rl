""" Utility functions 

"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
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
