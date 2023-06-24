# %%
"""
$ python experiments/08-ESN-STDP-MLops/nb\&flows/ops01/data_loading.py 
"""

import sys
import time
# sys.path += [".."]
# sys.path += [
#     ".", "..", "../..", "../../..", "../../../../",
#     "../../../../../", "experimentkit_in"]

from config import *
from experimentkit_in.generators.time_series import gen_lorenz
import src08.funx as src08_f


if __name__ == '__main__':
    params = argparse_config()


example_len = params['data']['example_len']
test_size = params['data']['test_size']
valid_size = params['data']['valid_size']
shift = params['data']['shift']

# --

t0 = time.time()
X_train, X_valid, X_test, y_train, y_valid, y_test = \
    src08_f.expt_generate_new_lorenz_data(
        example_len = example_len,
        test_size = test_size,
        valid_size = valid_size,
        recompute = True,
        ds_path=EXP_DATA_DIR/"ds_lorenz.pkl",
        shift = shift,  # forecasted delay
        s=12, r=30, b=2.700 # s=10, r=28, b=8/3
    )
te = time.time()
logger.info(f"elapsed time: {te-t0:.2f}s")


logger.info(
    f"X_train: {X_train.shape},\nX_valid: {X_valid.shape},\n"
    + f"\tX_test: {X_test.shape},\ny_train: {y_train.shape},\n"
    + f"\ty_valid: {y_valid.shape}\n, y_test: {y_test.shape}")

# store data
src08_f.pickle_save_dict(
    EXP_DATA_DIR/'ds.pkl',
    {
        'X_train': X_train,
        'X_valid': X_valid,
        'X_test': X_test,
        'y_train': y_train,
        'y_valid': y_valid,
        'y_test': y_test
    }
    )

# %%