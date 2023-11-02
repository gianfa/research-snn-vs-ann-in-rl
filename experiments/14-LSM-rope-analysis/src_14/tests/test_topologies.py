"""

$ pytest . --pdb
"""
import pytest
import torch
from src_14.topologies import gen_rope


def test_gen_rope():

    adj = gen_rope(n=3, m=3, radius=1, degree=2)
    exp = torch.tensor([
        [0., 1., 1.],
        [1., 0., 1.],
        [1., 1., 0.]])
    assert adj.equal(exp)

    adj = gen_rope(n=3, m=3, radius=1, degree=1)
    

def test_gen_rope__big_radius():

    with pytest.raises(ValueError):
        gen_rope(3, 3, radius=3, degree=2)