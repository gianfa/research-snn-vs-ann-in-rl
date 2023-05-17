# %%
""" Funx Tests

$ python -m pytest tests/stdp/test_funx.py -vv --pdb
"""
import collections
import sys
sys.path += ["../"]
sys.path += ["../.."] # good to test in jupyter
from typing import Iterable  # noqa

import pandas as pd
import pytest  # noqa
import torch  # noqa
from torch import nn

from stdp.funx import (ridge_regression_get_coeffs)  # noqa

from sklearn.datasets import make_regression

# %%



def test_ridge_regression():
    X, y, sk_coeffs = make_regression(
        n_samples=50,
        n_features=5,
        n_informative=5,
        n_targets=1,
        noise=5,
        coef=True,
        random_state=1
    )

    alpha = 0.5

    coeffs = ridge_regression_get_coeffs(
        torch.tensor(X), torch.tensor(y).unsqueeze(1)).flatten()

    sk_coeffs = torch.tensor(sk_coeffs)

    assert all(abs(coeffs - sk_coeffs)/sk_coeffs < 0.1)
    assert all(abs(coeffs - sk_coeffs)/coeffs < 0.1)
# %%
