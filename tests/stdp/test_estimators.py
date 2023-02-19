""" STDP Tests

$ python -m pytest tests/stdp/test_estimators.py -vv --pdb
"""
import collections
import sys
sys.path.append("../")
from typing import Iterable  # noqa

import pytest  # noqa
import torch  # noqa
from torch import nn

from stdp.estimators import define_spiking_cluster
from stdp.funx import (stdp_generate_dw_lookup, stdp_step, 
    model_get_named_layer_params, model_layers_to_weights,
    )  # noqa
from stdp.spike_collectors import nearest_pre_post_pair
from stdp.estimators import ESN

@pytest.fixture
def simple_pre_post_W_A():

    W = torch.tensor([
        [0, 0.5],
        [0.5, 0],
    ])
    """Weight matrix"""

    A = torch.tensor([
        [0, 1],
        [-1, 0],
    ])
    """Adjacency matrix"""

    return W, A


@pytest.fixture
def dw_time_lookup_40():
    return stdp_generate_dw_lookup(40)


def test_define_spiking_cluster():
    clust = define_spiking_cluster(
        tot_neurons = 100,
    )
    weights = clust['weights']
    in_ns = clust['input']
    out_ns = clust['output']
    
    assert weights.ndim == 2
    assert weights.shape[0] == weights.shape[1]
    assert type(in_ns) == type(out_ns) == list


def test_ESN_simple():
    input_size = 1
    hidden_size = 100
    output_size = 1
    spectral_radius = 0.9

    esn = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        spectral_radius=spectral_radius)
    criterion = nn.MSELoss()
    optimizer = torch.optim.LBFGS(esn.parameters(), lr=1, max_iter=20)
    epochs = 5

    def train(input, target):
        def closure():
            optimizer.zero_grad()
            output = esn(input)
            loss = criterion(output, target)
            loss.backward()
            return loss
        optimizer.step(closure)

    # One
    # TODO: verify, through assert
    for epoch in range(epochs):
        input = torch.randn(1000, 1)
        target = torch.sin(input)
        train(input, target)

    # Two
    X = torch.linspace(-5, 5, 1000).unsqueeze(1)
    out = esn(X)
    assert out.shape == torch.Size([output_size])


def extract_num(n: int = 1, max_int: int = 100):
    return torch.randint(0, max_int, (1, n)).flatten().tolist()


def test_ESN_simple__many_sizes():
    for _ in range(10):
        input_size = extract_num(1)[0]
        hidden_size = extract_num(1)[0]
        output_size = extract_num(1)[0]
        spectral_radius = 0.9

        esn = ESN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            spectral_radius=spectral_radius)

        X = torch.linspace(-5, 5, input_size).reshape(1, input_size)
        out = esn(X)
        assert out.shape == torch.Size([output_size])


def test_ESN_with_hidden_weights():
    input_size = 1
    hidden_size = 100
    output_size = 1
    spectral_radius = 0.9
    hidden_weights = torch.rand(hidden_size, hidden_size)
    
    # set zero positions to check later
    hidden_zeros_pos = [(0,4),(0,6)]
    hidden_weights[hidden_zeros_pos] = 0

    esn = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        hidden_weights=hidden_weights,
        spectral_radius=spectral_radius)

    input = torch.randn(1000, 1)
    out = esn(input)
    assert out.shape == torch.Size([output_size])


    # now check for the zero positions
    hidden_W = [param for name, param in esn.named_parameters() if name == 'W.weight'][0]
    hidden_W_zeros = torch.argwhere(hidden_W==0)
    assert torch.allclose(hidden_W_zeros.T, torch.tensor(hidden_zeros_pos))
    breakpoint()
