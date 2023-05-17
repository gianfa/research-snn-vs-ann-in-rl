""" STDP Tests

$ python -m pytest /Users/giana/Desktop/Github/Projects/research-snn-vs-ann-in-rl/experiments/ESN_ts/test_tryESN_1.py -vv --pdb -s
"""
import collections
import sys
sys.path.append("../")
from typing import Callable, Iterable, List, Tuple  # noqa

import pytest  # noqa
import torch  # noqa
from torch import nn
from torch.optim.optimizer import Optimizer


def test_try__ESN_1():
    import try__ESN_1




