"""
$ python -m pytest tests/test_viz.py -s --pdb
"""
import numpy as np

import pytest
from stdp.funx import (get_raster_from_spike_positions,
    get_spike_positions_from_tpre_tpost, get_raster_from_tpre_tpost)


def test_get_raster_from_spike_positions():
    """Test get_raster_from_spike_positions."""

    spike_positions_long = {
        0: [1, 3],
        2: [2, 4]
    }
    with pytest.raises(ValueError):
        get_raster_from_spike_positions(spike_positions_long, size=(1, 5))

    spike_positions_wide = {
        0: [1, 3],
        2: [2, 4]
    }
    with pytest.raises(ValueError):
        get_raster_from_spike_positions(spike_positions_wide, size=(3, 3))

    spike_positions = {
        0: [1, 3],
        2: [2, 4]
    }
    expected = [
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1]
    ]

    raster = get_raster_from_spike_positions(spike_positions, size=(3, 5))
    assert raster == expected


def test_get_spike_positions_from_tpre_tpost():
    """Test get_spike_positions_from_tpre_tpost."""

    tpre_tpost = [
        (0, 2),
        (2, 2),
        (4, 5)
    ]
    neurons = [4, 6]
    expected = {
        4: [0, 2, 4],
        6: [2, 5]
    }
    spk_pos = get_spike_positions_from_tpre_tpost(tpre_tpost, neurons)
    assert spk_pos == expected


def test_get_raster_from_tpre_tpost():
    tpre_tpost = [
        (0, 2),
        (2, 2),
        (4, 5)
    ]
    neurons = [4, 6]
    expected = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1]
    ]
    raster = get_raster_from_tpre_tpost(tpre_tpost, neurons)
    assert type(raster) == list
    assert raster == expected
