""" Configuration

"""
# %%
import itertools
import os
import sys
import time
# imports
from functools import partial
from pathlib import Path
from typing import List

import dataframe_image
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import snntorch as snn
import torch
import torch.nn as nn
from snntorch import spikegen
from snntorch import spikeplot as splt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path += ['../', '../../../']
# sys.path += [__file__, f"{__file__}/..", f"{__file__}/../.."]

import experimentkit_in as ek
from experimentkit_in.logger_config import setup_logger
import src_15

# %% Parameters

ROOT = Path("../../../").resolve()
EXP_DIR = Path("../").resolve()
EXP_DATA_DIR = EXP_DIR/"data"
EXP_REPORT_DIR = EXP_DIR/"reports"
RANDOMSEED = 42

assert ROOT.name == 'research-snn-vs-ann-in-rl'
assert EXP_DIR.name == '15-LSM-rope-complex_sig'

DTYPE = torch.float
DEVICE = "cpu"
# torch.device("cuda") \
#     if torch.cuda.is_available() else torch.device("cpu")

# %%

matplotlib.rcParams["figure.figsize"] = (18, 10)
matplotlib.rcParams["axes.titlesize"] = 21

np.random.seed(RANDOMSEED)

import matplotlib.pyplot as plt

# For publication
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 1.5
# Good contrast
# plt.rcParams['axes.prop_cycle'] = plt.cycler(
#     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

# plt.style.use("seaborn")  # more pleasant
pd.options.display.max_rows = 30
pd.options.display.max_columns = 20
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 999)

sns.set_context('paper')  # context for the print
sns.set_style('whitegrid')  # background style
# sns.set_palette('pastel')  #

# other
sns.set_context(
    rc={
        "font.size": 10,  # dict to override the context
        # "axes.titlesize": 22,
        # "axes.labelsize": 20,
        # "xtick.labelsize": 16,
        # "ytick.labelsize": 16,
    }
)

# utils

def text_header(txt, output_size=20):
    return "=" * 50, f"\n {txt.center(output_size)}\n", "=" * 50

def print_header(txt, output_size=20):
    print(text_header)

# %%

logger = setup_logger('experiment', log_level='DEBUG')
