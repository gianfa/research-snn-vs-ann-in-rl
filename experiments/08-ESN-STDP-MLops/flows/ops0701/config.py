import argparse
from pathlib import Path
import sys
# sys.path += ["..", "../..", "../../..", "../../../../", "../../../../../"]
sys.path += [
    ".", "experimentkit_in",
    "experiments/08-ESN-STDP-MLops/",
    "experiments/08-ESN-STDP-MLops/src08"
    "experiments/08-ESN-STDP-MLops/nb\&flows/",
]
from typing import Dict

import yaml

from experimentkit_in.logger_config import setup_logger

ROOT = Path('/Users/giana/Desktop/Github/Projects/research-snn-vs-ann-in-rl')
DATA_DIR = ROOT/'data'
EXP_DIR = ROOT/'experiments/08-ESN-STDP-MLops'
EXP_DATA_DIR = EXP_DIR/'data'
EXP_REPORT_DIR = EXP_DIR/'report'

SUBEXP_DATA_DIR = EXP_DATA_DIR/"exp-2/produced_data"

logger = setup_logger('ESN-STDP-Lorenz')

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