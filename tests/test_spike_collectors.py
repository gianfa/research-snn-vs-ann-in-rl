""" STDP Tests

$ python -m pytest tests/test_spike_collectors.py -vv --pdb
"""

# %%
import sys
sys.path.append("../")

import pytest  # noqa
from typing import List  # noqa
import torch  # noqa
from stdp.spike_collectors import nearest_pre_post_pair, all_to_all # noqa


def assert_collected_spks_are_expected(
    raster_2D: torch.Tensor,
    expected_spikes: List[tuple],
    collection_rule: callable,
) -> None:
    pre_raster = raster_2D[0, :]
    post_raster = raster_2D[1, :]
    spks = collection_rule(pre_raster, post_raster)
    assert set(spks) == set(expected_spikes)


def test__all_to_all():

    raster_nospks = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    expected_spks = []

    pre_raster = raster_nospks[0, :]
    post_raster = raster_nospks[1, :]
    spks = all_to_all(pre_raster, post_raster)
    assert set(spks) == set(expected_spks)

    # NOTE: This is quite interesting
    raster_nospks = torch.tensor([
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0]
    ])
    expected_spks = [(1, 3), (3, 1)]

    pre_raster = raster_nospks[0, :]
    post_raster = raster_nospks[1, :]
    spks = all_to_all(pre_raster, post_raster)
    assert set(spks) == set(expected_spks)

    #
    raster_1t = torch.tensor([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])
    expected_spks = [(1, 2)]

    pre_raster = raster_1t[0, :]
    post_raster = raster_1t[1, :]
    spks = all_to_all(pre_raster, post_raster)
    assert set(spks) == set(expected_spks)

    #
    raster_2t = torch.tensor([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]
    ])
    expected_spks = [(1, 3)]

    pre_raster = raster_2t[0, :]
    post_raster = raster_2t[1, :]
    spks = all_to_all(pre_raster, post_raster)
    assert set(spks) == set(expected_spks)

    raster_after1t = torch.tensor([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])
    expected_spks = [(3, 2)]

    pre_raster = raster_after1t[0, :]
    post_raster = raster_after1t[1, :]
    spks = all_to_all(pre_raster, post_raster)
    assert set(spks) == set(expected_spks)

    #
    raster_both_1t = torch.tensor([
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])
    expected_spks = [(1, 2), (3, 2)]

    pre_raster = raster_both_1t[0, :]
    post_raster = raster_both_1t[1, :]
    spks = all_to_all(pre_raster, post_raster)
    assert set(spks) == set(expected_spks)

    # Multiple #

    #
    raster_both_2x_1dt = torch.tensor([
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0]
    ])
    expected_spks = [
        (1, 2),
        (1, 4),
        (3, 2),
        (3, 4)
    ]

    pre_raster = raster_both_2x_1dt[0, :]
    post_raster = raster_both_2x_1dt[1, :]
    spks = all_to_all(pre_raster, post_raster)
    assert set(spks) == set(expected_spks)

    #
    raster_both_2x_1dt = torch.tensor([
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0]
    ])
    expected_spks = [
        (1, 2),
        (1, 4),
        (3, 2),
        (3, 4),
        (4, 2)
    ]

    pre_raster = raster_both_2x_1dt[0, :]
    post_raster = raster_both_2x_1dt[1, :]
    spks = all_to_all(pre_raster, post_raster)
    assert set(spks) == set(expected_spks)


def test__nearest_pre_post_pair():

    raster_nospks = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    expected_spks = []

    pre_raster = raster_nospks[0, :]
    post_raster = raster_nospks[1, :]
    spks = nearest_pre_post_pair(pre_raster, post_raster)
    assert set(spks) == set(expected_spks)

    # NOTE: This is quite interesting
    raster_nospks = torch.tensor([
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0]
    ])
    expected_spks = [(3, 1)]

    pre_raster = raster_nospks[0, :]
    post_raster = raster_nospks[1, :]
    spks = nearest_pre_post_pair(pre_raster, post_raster)
    assert set(spks) == set(expected_spks)

    #
    raster_1t = torch.tensor([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])
    expected_spks = [(1, 2)]

    pre_raster = raster_1t[0, :]
    post_raster = raster_1t[1, :]
    spks = nearest_pre_post_pair(pre_raster, post_raster)
    assert set(spks) == set(expected_spks)

    #
    raster_2t = torch.tensor([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]
    ])
    expected_spks = [(1, 3)]

    pre_raster = raster_2t[0, :]
    post_raster = raster_2t[1, :]
    spks = nearest_pre_post_pair(pre_raster, post_raster)
    assert set(spks) == set(expected_spks)

    raster_after1t = torch.tensor([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])
    expected_spks = [(3, 2)]

    pre_raster = raster_after1t[0, :]
    post_raster = raster_after1t[1, :]
    spks = nearest_pre_post_pair(pre_raster, post_raster)
    assert set(spks) == set(expected_spks)

    #
    raster_both_1t = torch.tensor([
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])
    expected_spks = [(1, 2)]

    pre_raster = raster_both_1t[0, :]
    post_raster = raster_both_1t[1, :]
    spks = nearest_pre_post_pair(pre_raster, post_raster)
    assert set(spks) == set(expected_spks)

    # Multiple #

    #
    raster_both_2x_1dt = torch.tensor([
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0]
    ])
    expected_spks = [
        (1, 2),
        (3, 4)
    ]

    pre_raster = raster_both_2x_1dt[0, :]
    post_raster = raster_both_2x_1dt[1, :]
    spks = nearest_pre_post_pair(pre_raster, post_raster)
    assert set(spks) == set(expected_spks)

    #
    raster_both_2x_1dt = torch.tensor([
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0]
    ])
    expected_spks = [
        (1, 2),
        (3, 4)
    ]

    pre_raster = raster_both_2x_1dt[0, :]
    post_raster = raster_both_2x_1dt[1, :]
    spks = nearest_pre_post_pair(pre_raster, post_raster)
    assert set(spks) == set(expected_spks)
