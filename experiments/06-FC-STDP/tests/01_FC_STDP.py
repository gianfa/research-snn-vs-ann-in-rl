"""

$ python -m pytest experiments/06-FC-STDP/tests/01_FC_STDP.py -vv --qqpdb
"""
import sys
sys.path += ['.', '..', './experiments/06-FC-STDP/nb&flows']

import os
def test_01_FC_STDP_():
    print(os.getcwd())
    print(os.listdir())
    import _01_FC_STDP

    # ppp = np.stack(pre_post)
    # raster.shape

    # TODO:
    # - raster must be 2d, here is 3d.
    #  [num_steps, batch_size, n_output]


    # STDP:
    # - take fc2.weights
    # - take the net raster (output): maybe we need the
    #   specific internal layer's one.