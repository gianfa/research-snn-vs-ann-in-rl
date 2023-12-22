"""


Usage
-----
from /experiments/08-ESN-STDP-MLops/01-forecast-lorenz


"""
# %%
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import yaml

ROOT = (Path(__file__)/"../../../../../")

assert ROOT.resolve().name == 'research-snn-vs-ann-in-rl'

EXP_DIR = Path(__file__).parent.parent
EXP_NAME = EXP_DIR.name
EXP_DATA_DIR = EXP_DIR/'data'
EXP_REPORT_DIR = EXP_DIR/'report'
EXP_FLOW_DIR = EXP_DIR/'flow'
ROOT_PATH = str(ROOT.resolve())

sys.path += [
    ".", ROOT_PATH, str(EXP_DIR.parent.resolve())
]

import experimentkit_in as ek

# %%
logger = ek.logger_config.setup_logger(EXP_NAME)

def load_yaml(fpath: str = None) -> Dict[str, object]:
    params = None
    if fpath is None:
        fpath = EXP_DIR/'params.yaml'
    with open(fpath, "r") as stream:
        try:
            params = yaml.safe_load(stream)
            logger.info(params)
            return params
        except yaml.YAMLError as e:
            logger.error(e)

def argparse_config():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", dest="config", required=False)
    args = arg_parser.parse_args()

    params = load_yaml(fpath=args.config)
    return params