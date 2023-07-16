
# %%
""" ESN Tests

$ 
"""
import collections
import sys
sys.path += ["..", "../..", "../../.."]
from typing import Callable, Iterable, List, Tuple  # noqa

import matplotlib.pyplot as plt
import torch  # noqa
from torch import nn
from torch.optim.optimizer import Optimizer
from sklearn.datasets import make_regression
from stdp.estimators import ESN

# # Data Generation
x, y, sk_coeffs = make_regression(
    n_samples=50,
    n_features=5,
    n_informative=5,
    n_targets=1,
    noise=5,
    coef=True,
    random_state=1
)
X = torch.tensor(x)
Y = torch.tensor(y).unsqueeze(1)

# %%

# # Define ESN
input_size = X.shape[1]
hidden_size = 100
output_size = Y.shape[1]
spectral_radius = 0.9

# -
esn = ESN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    spectral_radius=spectral_radius)
esn.train(X, Y, epochs=5, lr=1e-2, v=True)

plt.show()
# plt.scatter(x[:, 0], y)
# %%