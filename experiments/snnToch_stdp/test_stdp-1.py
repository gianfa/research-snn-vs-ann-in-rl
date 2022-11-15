"""

$ python -m pytest -k test_stdp -vv -s --pdb

References
----------

Papers
https://www.scienceopen.com/document_file/0f0e3391-68df-4b06-b08a-27b79c0869b2/PubMedCentral/0f0e3391-68df-4b06-b08a-27b79c0869b2.pdf

Web
1: https://stackoverflow.com/questions/54995306/how-to-implement-stdp-in-tensorflow
3. https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial4.html

Code
1. https://brian2.readthedocs.io/en/stable/resources/tutorials/2-intro-to-brian-synapses.html
2. https://brian2.readthedocs.io/en/stable/examples/synapses.STDP.html
4. https://github.com/guillaume-chevalier/Spiking-Neural-Network-SNN-with-PyTorch-where-Backpropagation-engenders-STDP

"""
# %%
import sys
sys.path.append("../")

# imports
import os  # noqa
from typing import Iterable  # noqa
import snntorch as snn  # noqa
from snntorch import surrogate  # noqa
from snntorch import backprop  # noqa
from snntorch import functional as SF  # noqa
from snntorch import utils  # noqa
from snntorch import spikeplot as splt  # noqa
from snntorch import spikegen  # noqa
from tut_utils import * # noqa

import torch  # noqa
import torch.nn as nn  # noqa
from torch.optim import Optimizer  # noqa
from torch.utils.data import DataLoader  # noqa
from torchvision import datasets, transforms  # noqa
import torch.nn.functional as F  # noqa

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import itertools  # noqa

# %%
from torch import Tensor  # noqa


class STDPFractionalLoss(nn.Module):
    """
    Notes
    -----
    unsupervised
    target is the dw_in
    """
    def __init__(self, *args):
        super(STDPFractionalLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        dw_in = target
        dw_out = input
        return ((dw_in - dw_out)/dw_in).mean()


model = nn.Linear(2, 2)
dw_in = torch.tensor([1, 2, 3, 3, 2])
dw_out = torch.tensor([1, 1.5, 2, 3, 1])

stdp_floss = STDPFractionalLoss()
loss = stdp_floss(dw_out, dw_in)
print(loss)
# loss.backward()

# %%


def stdp_abbott(
    A_plus: float, A_minus: float,
        tau_plus: float, tau_minus: float, delta_t: Iterable[int]) -> float:
    dw = torch.zeros(len(delta_t))
    # TODO: to multiply by w_max
    dw[delta_t <= 0] = A_plus * torch.exp(delta_t[delta_t <= 0]/tau_plus)
    dw[delta_t > 0] = -A_minus * torch.exp(-delta_t[delta_t > 0]/tau_minus)
    return dw

# def stdp(
#     n_
#     A_plus: float, A_minus: float,
#         tau_plus: float, tau_minus: float, delta_t: Iterable[int]) -> float:
#     dw = torch.zeros(len(delta_t))
#     # TODO: to multiply by w_max
#     dw[delta_t <= 0] = A_plus * torch.exp(delta_t[delta_t <= 0]/tau_plus)
#     dw[delta_t > 0] = -A_minus * torch.exp(-delta_t[delta_t > 0]/tau_minus)
#     return dw


class STDP(Optimizer):
    """
    """

    def __init__(
        self,
        params: dict,
        delta_t: torch.Tensor,
        A_plus: float = 5e-3,
        A_minus: float = 5e-3,
        tau_plus: float = 20e-3,
        tau_minus: float = 20e-3,
    ):
        defaults = dict(  # <--`defaults` is a needed variable
            A_plus=A_plus,
            A_minus=A_minus,
            tau_plus=tau_plus,
            tau_minus=tau_minus,
            delta_t=delta_t,
        )
        super(STDP, self).__init__(params, defaults)
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

    @torch.no_grad()
    def step(self):
        # loss = None
        for group in self.param_groups:
            W = group['params'][0].data  # weigths matrix
            dW = stdp_abbott(
                A_plus=group['A_plus'],
                A_minus=group['A_minus'],
                tau_plus=group['tau_plus'],
                tau_minus=group['tau_minus'],
                delta_t=group['delta_t'])
            group['params'][0].data = W + dW


# def test_one():
l1 = nn.Linear(6, 6)
loss_fn = nn.CrossEntropyLoss()
optimizer = STDP(l1.parameters())

n_batches = 20
x_batches = torch.arange(6*n_batches).reshape(n_batches, 6).float()
y_batches = torch.tensor([1, 0, 1, 0, 1, 0]).repeat(n_batches, 1).float()
print(f"x.shape {x_batches.shape}; y.shape {y_batches.shape}")
x = x_batches[0]
y = y_batches[0]

optimizer.zero_grad()
output = l1(x)
loss = loss_fn(output, y)
loss.backward()
print(list(l1.parameters()))
optimizer.step()
print(list(l1.parameters()))