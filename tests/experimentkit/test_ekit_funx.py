# %%
""" [INCOMPLETE, DEV]

python -m pytest tests/experimentkit/test_funx.py -vv --pdb
"""
import pytest
import sys
sys.path += ["../.."] # good to test in jupyter

import matplotlib.pyplot as plt

from experimentkit_in.monitor import Monitor

@pytest.mark.skip(reason="postponed: needs IPython available")
def test_Monitor__init():
    import random
    n = 5
    x = range(n)
    y = [random.random()*10 for _ in range(n)]
    monitor = Monitor(x=x, y=y)

@pytest.mark.skip(reason="postponed: needs IPython available")
def test_Monitor__update():
    import random

    n = 5
    x = list(range(n))
    y = [random.random() * 10 for _ in range(n)]
    monitor = Monitor(x=x, y=y)

    for _ in range(10):
        y += [random.random()]
        x = list(range(len(y)))
        monitor.update(x=x, y=y)




# %%
