"""
$ python experiments/08-ESN-STDP-MLops/nb\&flows/ops01/data_loading.py 
"""

from typing import List
import time
# sys.path += [".."]
# sys.path += [
#     ".", "..", "../..", "../../..", "../../../../",
#     "../../../../../", "experimentkit_in"]


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch

from config import *
from experimentkit_in.generators.time_series import gen_lorenz
from stdp.estimators import BaseESN
from stdp import funx as stdp_f
import src08.funx as src08_f

if __name__ == '__main__':
    params = argparse_config()


example_len = params['data']['example_len']
test_size = params['data']['test_size']
valid_size = params['data']['valid_size']
shift = params['data']['shift']

# --

# Data Loading
ds = src08_f.pickle_load(EXP_DATA_DIR/'ds.pkl')

X_train = ds['X_train']
X_valid = ds['X_valid']
X_test = ds['X_test']
y_train = ds['y_train']
y_valid = ds['y_valid']
y_test = ds['y_test']


# 

# %% STDP-Execute
""" Many trials implementing STDP ESN weights update

Trial steps
-----------
1. Initialize the ESN.
2. Perform the STDP Update.

STDP Update
-----------
1. Train the ESN and get the network state history from training
2. Compute the new Reservoir weights, new_W, applying STDP to the states of
    the last 20 steps.
3. Replace the Reservoir weights with the new ones just computed.

"""


t0 = time.time()
th = 0  # spike threshold
perf_hist_nonstdp = {'mse': [], 'mae': [], 'r2': []}

perf_hist_after_stdp = {'mse': [], 'mae': [], 'r2': []}
"""Mean performance history of STDP optimisation
every element is the last performance along all the optimisation steps
"""

perf_hist_stdp_inner: Dict[str, List] = {'mse': [], 'mae': [], 'r2': []}
"""Performance history of STDP optimisasion step
every element is the performance at the specific optimisation step
"""

perf_hist_stdp = []

ds_path = EXP_DATA_DIR/f"ds_lorenz-{example_len}.pkl"

X_train, X_valid, X_test, y_train, y_valid, y_test = \
    src08_f.expt_generate_new_lorenz_data(
        example_len = 10000,
        test_size = 0.2,
        valid_size = 0.15,
        recompute = False,
        ds_path = ds_path,
        shift = 15,  # forecasted delay
        s=12, r=30, b=2.700
    )

W_hist_nonstdp = []
W_hist_stdp =  []

n_trials: int = params['experiment']['n_trials']
n_STDP_steps: int = params['experiment']['n_STDP_steps']

input_size = X_train.shape[1]
reservoir_size = params['model']['reservoir_size']
output_size = y_train.shape[1]
STDP_steps_scope = params['experiment']['STDP_steps_scope']
verbose = False
t0 = time.time()
for i in range(n_trials):
    logger.info(f"Trial #{i}")

    esn = BaseESN(
        input_size,
        reservoir_size,
        output_size,
        connections = (stdp_f.generate_simple_circle_connections_mask(
            (reservoir_size, reservoir_size)) > 0).int(),
        decay = params['model']['decay']
    )

    # Non-STDP
    state_hist = esn.train(X_train.float(), y_train.float())
    y_out = esn.predict(X_valid.float())
    y_pred = y_out
    
    report = src08_f.evaluate_regression_report(y_valid, y_pred)
    for m_name, m_val in report.items():
        if m_name not in perf_hist_nonstdp:
            perf_hist_nonstdp[m_name] = []
        perf_hist_nonstdp[m_name].append(m_val)
    
    W_hist_nonstdp.append(esn.W.data)
    
    # STDP Update
    perf_hist_stdp_inner = {}
    W_hist_stdp.append([])
    for epoch in range(n_STDP_steps):
        t_i = time.time()

        # STDP
        raster = (state_hist > th).to(int)[:, -STDP_steps_scope:]
        # update hidden connections
        reservoir_weights = esn.W.clone()
        reservoir_connections = esn.connections

        # print(layer.weight.mean())
        new_W = stdp_f.stdp_step(
            reservoir_weights,
            connections=reservoir_connections,
            raster=raster,
            spike_collection_rule = stdp_f.all_to_all,
            dw_rule = "sum",
            max_delta_t=4,
        )
        # Normalize weights
        # new_W /= new_W.max()
        new_W = ((new_W / new_W.abs().max()) * 2 - 1)
        if epoch % 2 == 0:
            new_W = (0.5 * new_W + \
                torch.randn_like(esn.W) * 0.5)
        new_W *= esn.connections

        # ensure weights matrix reflects the connections
        n_exceding_connections = (
            (new_W != 0).to(torch.int) - \
                (reservoir_connections != 0).to(torch.int)
            ).sum()
        assert n_exceding_connections == 0, f"{n_exceding_connections}"
        
        # Update ESN weights
        esn.W = new_W
        assert esn.W.equal(new_W)
        W_hist_stdp[-1].append(esn.W.data)

        if verbose: logger.info(f"STDP executed in {time.time() - t_i:.0f}s")

        # # Retrain after STDP and evaluate
        state_hist = esn.train(X_train.float(), y_train.float())
        y_out = esn.predict(X_valid.float())
        y_pred = y_out
        
        report = src08_f.evaluate_regression_report(y_valid, y_pred)
        for m_name, m_val in report.items():
            if m_name not in perf_hist_stdp_inner:
                perf_hist_stdp_inner[m_name] = []
            perf_hist_stdp_inner[m_name].append(m_val)
        
    # perf_hist_stdp.append(
    #     {k: np.mean(v) for k, v in perf_hist_stdp_inner.items()})
    perf_hist_stdp_inner = {
        k: torch.Tensor(v) for k, v in perf_hist_stdp_inner.items()}
    perf_hist_stdp.append(perf_hist_stdp_inner)

    for m_name, m_hist in perf_hist_stdp_inner.items():
            if m_name not in perf_hist_stdp_inner:
                perf_hist_after_stdp[m_name] = []
            perf_hist_after_stdp[m_name].append(m_hist[-1].item())

    logger.info(f"t: {time.time() - t0:.0f}s")
logger.info(f"elapsed time: {time.time() - t0:.2f}s")

logger.info(f"STDP performance stats:")
perf_stats_before = pd.DataFrame(perf_hist_nonstdp).describe()
logger.info("\n-- BEFORE --")
logger.info(f"\n{perf_stats_before}")

logger.info("\n-- AFTER STDP --")
perf_stats_after = pd.DataFrame(perf_hist_after_stdp).describe()
logger.info(f"\n{perf_stats_after}")

src08_f.pickle_save_dict(
    EXP_DATA_DIR/'perf_hist_nonstdp.pkl', perf_hist_nonstdp)
src08_f.pickle_save_dict(
    EXP_DATA_DIR/'perf_hist_after_stdp.pkl', perf_hist_after_stdp)
src08_f.pickle_save_dict(
    EXP_DATA_DIR/'perf_hist_stdp.pkl', {'perf_hist_stdp': perf_hist_stdp})
