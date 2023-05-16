""" STDP Tests

$ python -m pytest tests/stdp/test_stdp.py -vv --pdb
"""
import collections
import sys
sys.path += ["..", "../..", "../../.."]
from typing import Iterable  # noqa

import pytest  # noqa
import torch  # noqa
from torch import nn

from stdp.funx import (stdp_generate_dw_lookup, stdp_step, 
    model_get_named_layer_params, model_layers_to_weights)  # noqa
from stdp.spike_collectors import nearest_pre_post_pair

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


def test_stdp_generate_dw_lookup():
    max_dt = 5
    T_lu = stdp_generate_dw_lookup(max_dt)
    ks = list(T_lu.keys())
    expected = list(range(-max_dt, max_dt+1))
    assert ks == expected


def test_stdp__2n_prepost_1dt(simple_pre_post_W_A, dw_time_lookup_40):

    W, A = simple_pre_post_W_A
    T_lu = dw_time_lookup_40

    raster_1t = torch.tensor([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])
    expected_W_1t = torch.tensor([
        [0, W[0, 1] + T_lu[-1]],
        [W[1, 0] + T_lu[+1], 0]
    ])

    W_1t = stdp_step(
        weights=W,
        connections=A,
        raster=raster_1t,
        dw_rule="sum",
        bidirectional=True,
        max_delta_t=20,
        inplace=False,
        v=True
    )

    assert torch.allclose(W_1t, expected_W_1t)


def test_stdp__2n_prepost_0dt(simple_pre_post_W_A, dw_time_lookup_40):

    W, A = simple_pre_post_W_A

    raster_1t = torch.tensor([
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0]
    ])

    W_0t = stdp_step(
        weights=W,
        connections=A,
        raster=raster_1t,
        dw_rule="sum",
        bidirectional=True,
        max_delta_t=20,
        inplace=False,
        v=True
    )

    expected_W_0t = W
    assert torch.allclose(W_0t.float(), expected_W_0t.float(), atol=1e-4)


def test_stdp__2n_prepost_2dt(simple_pre_post_W_A, dw_time_lookup_40):

    W, A = simple_pre_post_W_A
    T_lu = dw_time_lookup_40

    raster_2t = torch.tensor([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]
    ])

    W_1t = stdp_step(
        weights=W,
        connections=A,
        raster=raster_2t,
        dw_rule="sum",
        bidirectional=True,
        max_delta_t=20,
        inplace=False,
        v=True
    )

    expected_W_2t = torch.tensor([
        [0, W[0, 1] + T_lu[-2]],
        [W[1, 0] + T_lu[+2], 0]
    ])

    assert torch.allclose(W_1t, expected_W_2t)


def test_stdp__2n_prepost_dt_gt_maxdt(simple_pre_post_W_A, dw_time_lookup_40):

    W, A = simple_pre_post_W_A
    T_lu = dw_time_lookup_40

    raster_2t = torch.tensor([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]
    ])

    msg = "With max distance between spikes greater than max_dt, "
    W_1t = stdp_step(
        weights=W,
        connections=A,
        raster=raster_2t,
        dw_rule="sum",
        bidirectional=True,
        max_delta_t=2,
        inplace=False,
        v=True
    )

    expected_W_2t = torch.tensor([
        [0, W[0, 1] + T_lu[-2]],
        [W[1, 0] + T_lu[+2], 0]
    ])

    assert torch.allclose(W_1t, expected_W_2t), f"{msg} should not raise error"


def __SKIP__test_stdp__2n_prepost_preburst_2t(simple_pre_post_W_A, dw_time_lookup_40):
    """
    nearest spike(pre) - nearest spike(post)
    consider only post spikes - take a look at your left and right nearest
    spikes on the pre
    """

    W, A = simple_pre_post_W_A
    T_lu = dw_time_lookup_40

    raster_2t_preburst = torch.tensor([
        [1, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 1, 0]
    ])

    # raster_2t_preburst = torch.tensor([
    #     [1, 1, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0]
    # ])

    # raster_2t_preburst = torch.tensor([
    #     [1, 1, 1, 0, 0, 1],  # neuron a
    #     [0, 0, 0, 0, 1, 0]   # neuron b
    # ])

    W_2t_preburst = stdp_step(
        weights=W,
        connections=A,
        raster=raster_2t_preburst,
        spike_collection_rule=nearest_pre_post_pair,
        dw_rule="sum",
        bidirectional=True,
        max_delta_t=20,
        inplace=False,
        v=True
    )

    expected_W_2t_preburst = torch.tensor([
        [0, W[0, 1] + T_lu[-2]],
        [W[1, 0] + T_lu[+2], 0]
    ])

    assert torch.allclose(W_2t_preburst, expected_W_2t_preburst)


    raster_2t_postburst = torch.tensor([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0]
    ])


    raster_v_seq = torch.tensor([
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])
    """V seq
    Will this be TLP or LTD?
    """


    raster_2t_preburst = torch.tensor([
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0]
    ])


def test_model_get_named_layer_params():
    
    net = torch.nn.Sequential(
        collections.OrderedDict([
            ("lin1", nn.Linear(10, 9)),
            ("lin2", nn.Linear(9, 5)),
            ("lin3", nn.Linear(5, 10)),
            ("lin4", nn.Linear(10, 10)),
        ])
    )

    lin1_par = model_get_named_layer_params(net, "lin1")
    assert type(lin1_par) == torch.nn.parameter.Parameter
    assert list(lin1_par.shape) == [9, 10]
    lin2_par = model_get_named_layer_params(net, "lin2")
    assert type(lin2_par) == torch.nn.parameter.Parameter
    assert list(lin2_par.shape) == [5, 9]


# function currently not used
def __SKIP__test_model_layers_to_weights():

    net = torch.nn.Sequential(
        collections.OrderedDict([
            ("lin1", nn.Linear(10, 9)),
            ("lin2", nn.Linear(9, 5)),
            ("lin3", nn.Linear(5, 10)),
            ("lin4", nn.Linear(10, 10)),
        ])
    )

    weights = model_layers_to_weights(net, ["lin2", "lin3"])






# %%
