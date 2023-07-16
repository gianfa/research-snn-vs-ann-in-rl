import sys
sys.path += ["..", "../..", "../../.."]

import time
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd # noqa
import seaborn as sns
import snntorch as snn  # noqa
from snntorch import spikeplot as splt  # noqa
from snntorch import spikegen  # noqa
from sklearn.metrics import ( # noqa
    accuracy_score, precision_score, f1_score, confusion_matrix)
import torch  # noqa
import torch.nn as nn  # noqa
from torch.utils.data import Dataset, DataLoader, TensorDataset  # noqa
from torchvision import datasets, transforms  # noqa
import tqdm

from experimentkit_in.visualization import plot_n_examples
from experimentkit_in.metricsreport import MetricsReport
from experimentkit_in.metricsreport import MetricsReport
from experimentkit_in.generators.time_series import *
from experimentkit_in.monitor import Monitor
from experimentkit_in.funx import pickle_save_dict
from stdp.funx import stdp_step


def batch_accuracy(net, data, targets, batch_size, layer_name=None):
    # From snnTorch examples
    output, _ = net(data.view(batch_size, -1))
    # take idx of the max firing neuron, as the y_pred
    output = output[layer_name] if layer_name else output
    _, idx = output.sum(dim=0).max(dim=1)
    acc = np.mean((targets == idx).detach().cpu().numpy())
    return acc


def print_batch_accuracy(data, targets, train=False):
    acc = batch_accuracy(data, targets)

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")


def training_bar_init(total, **kwargs):
    import warnings
    warnings.filterwarnings('ignore')
    pbar = tqdm.tqdm(total=total, **kwargs)
    return pbar


def training_bar_update(
        arg_dict: dict, pbar: tqdm.std.tqdm, progress: float) -> None:
    desc = "|".join([
        f"{k}:{v.__round__(2) if type(v) == float else v}"
            for k, v in arg_dict.items()])
    pbar.update(progress)
    pbar.set_description(desc)
    pbar.refresh()
    return pbar



def print_vars_table(arg_dict: dict, print_title: bool=False) -> None:
    if print_title:
        title = "|" + "|".join(
            [f"{arg_name:^10}" for arg_name in arg_dict.keys()]) + "|"
        title += "\n|" + "|".join(
            [f"{'----------':^10}" for arg_name in arg_dict.keys()]) + "|"
        print(title)

    values = "|" + "|".join(
            [f"{arg_value:^10}" for arg_value in arg_dict.values()]) + "|"
    print(values)
    return None


def print_batch_accuracy(net, data, targets, batch_size, train=False):
    # https://github.com/jeshraghian/snntorch/blob/master/docs/tutorials/tutorial_5.rst#71-accuracy-metric
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")


def train_printer(
    epoch, iter_counter,
    test_data, test_targets,
    loss_hist, test_loss_hist,
    data, targets,
    ):
    # https://github.com/jeshraghian/snntorch/blob/master/docs/tutorials/tutorial_5.rst#71-accuracy-metric
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

