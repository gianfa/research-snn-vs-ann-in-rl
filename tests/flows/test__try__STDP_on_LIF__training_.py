"""

$ python -m pytest tests/flows/test__try__STDP_on_LIF__training_.py -vv --pdb
$ python flows/test__try__STDP_on_LIF__training_.py
"""
import sys
sys.path += [ '.']

import os
def __SKIP__test_flows_try_STDP_on_LIF_training_():
    print(os.chdir("flows"))
    print(os.getcwd())
    print(os.listdir())
    import try__STDP_on_LIF__training_

    # ppp = np.stack(pre_post)
    # raster.shape

    # TODO:
    # - raster must be 2d, here is 3d.
    #  [num_steps, batch_size, n_output]


    # STDP:
    # - take fc2.weights
    # - take the net raster (output): maybe we need the
    #   specific internal layer's one.