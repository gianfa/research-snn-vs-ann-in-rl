""" Tests for LIF-STDP-ts

pytest ./experiments/LIF-STDP-ts/tests/test_class_ts.py -vv --pdb
"""
import sys
sys.path += ['experiments/LIF-STDP-ts', '.', "..", "../.."]

import os
def test_class_ts():
    print(os.getcwd())
    print(os.listdir())
    import class_ts__dev

    # ppp = np.stack(pre_post)
    # raster.shape