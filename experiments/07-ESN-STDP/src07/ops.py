"""Operations for executing the experiments"""


from copy import deepcopy
import itertools
import os
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import wandb

sys.path += ["..", "../..", "../../.."]
from experimentkit_in.visualization import get_cmap_colors
from experimentkit_in.funx import pickle_save_dict, pickle_load, load_yaml
from experimentkit_in.metricsreport import MetricsReport
from experimentkit_in.generators.time_series import gen_lorenz
from stdp.estimators import BaseESN
from stdp import funx as stdp_f

import src07.funx as src07_f

ROOT = Path("../../../")
# ROOT = Path("../../")
DATA_DIR = ROOT / "data"
EXP_DIR = ROOT / "experiments/07-ESN-STDP"
EXP_DATA_DIR = EXP_DIR / "data"
EXP_REPORT_DIR = EXP_DIR / "report"
EXP_NAME = "exp-1"
EXP_RESULTS_DIR = EXP_DATA_DIR / EXP_NAME

assert EXP_DIR.exists(), f"CWD: {Path.cwd()}\nROOT: {ROOT.absolute()}"


def generate_data():
    