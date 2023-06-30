""" Lorenz Forecast classification with ESN + STDP



Premises and observations:


Exp Observations:
    1. It is prone to exploding weights since STDP will amplify indefinitely.
        -> Normalization before update.
    2. 

        
Ref:
1. SCHAETTI, Nils; SALOMON, Michel; COUTURIER, Raphaël. Echo state networks-based reservoir computing for mnist handwritten digits recognition. In: 2016 IEEE Intl Conference on Computational Science and Engineering (CSE) and IEEE Intl Conference on Embedded and Ubiquitous Computing (EUC) and 15th Intl Symposium on Distributed Computing and Applications for Business Engineering (DCABES). IEEE, 2016. p. 484-491.
"""
# %%
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch

sys.path += ["..", "../..", "../../.."]
from experimentkit_in.funx import pickle_save_dict, pickle_load
from experimentkit_in.metricsreport import MetricsReport
from experimentkit_in.generators.time_series import gen_lorenz
from stdp.estimators import BaseESN
from stdp import funx as stdp_f

import src07.funx as src07_f

ROOT = Path('../../../')
DATA_DIR = ROOT/'data'
EXP_DIR = ROOT/'experiments/07-ESN-STDP'
EXP_DATA_DIR = EXP_DIR/'data'
EXP_REPORT_DIR = EXP_DIR/'report'

assert EXP_DIR.exists(), \
    f"CWD: {Path.cwd()}\nROOT: {ROOT.absolute()}"

# %% Data Loading

ds_path = EXP_DATA_DIR/"ds_lorenz.pkl"
if  not ds_path.exists():
    ds = gen_lorenz(n_steps=10000, s=12, r=30, b=2.700)
    pickle_save_dict(ds_path, ds)
else:
    ds = pickle_load(ds_path)


shift = 12
X = ds[:-shift]
y = ds[shift:, 0]

print(X.shape, y.shape)
assert X.shape[0] == y.shape[0]

# %% Train/Test split

test_size = 0.2
valid_size = 0.15

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, shuffle=False)

valid_size = 0.15 / (1 - test_size)
X_train, X_valid, y_train, y_valid = \
    train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

X_train = torch.tensor(X_train, dtype=float)
X_valid = torch.tensor(X_valid, dtype=float)
X_test = torch.tensor(X_test, dtype=float)
y_train = torch.tensor(y_train, dtype=float).unsqueeze(1)
y_valid = torch.tensor(y_valid, dtype=float).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=float).unsqueeze(1)

print(
    X_train.shape, y_train.shape,
    X_valid.shape, y_valid.shape,
    X_test.shape, y_test.shape)

# %% Model definition

input_size = X_train.shape[1]
reservoir_size = 100
output_size = y_train.shape[1]
esn = BaseESN(
    input_size,
    reservoir_size,
    output_size,
    connections = (stdp_f.generate_simple_circle_connections_mask(
        (reservoir_size, reservoir_size)) > 0).int()
)

# Training

states = esn.train(X_train.float(), y_train.float())

# %% 1 Trial

y_out = esn.predict(X_valid.float())
y_pred = y_out
print(y_pred.shape)

# %% 1 Trial: Plot

src07_f.plot_evaluate_regression_report(y_valid, y_pred)

# %% Many Trials: Performance distribution
hist = {}

t0 = time.time()
for i in range(100):
    # Define model
    esn = BaseESN(
        input_size,
        reservoir_size,
        output_size,
        connections = (stdp_f.generate_simple_circle_connections_mask(
            (reservoir_size, reservoir_size)) > 0).int()
    )

    states = esn.train(X_train.float(), y_train.float())
    y_out = esn.predict(X_valid.float())
    y_pred = y_out
    report = src07_f.evaluate_regression_report(y_valid, y_pred)
    for m_name, m_val in report.items():
        if m_name not in hist:
            hist[m_name] = []
        hist[m_name].append(m_val)
print(f"Executed in {time.time()-t0:.0f}s")

# Plot
fig, axs = plt.subplots(1, len(hist))

sns.histplot(hist['mae'], ax=axs[0])
axs[0].set(title='MAE', yscale='log')

sns.histplot(hist['mse'], ax=axs[1])
axs[1].set(title='MSE', yscale='log')

sns.histplot(hist['r2'], ax=axs[2])
axs[2].set(title='R2', yscale='log')

perf_stats_before = pd.DataFrame(hist).describe()
print(perf_stats_before)


# %% STDP-Execute

t0 = time.time()
th = 0
hist = {}
W_hist = [esn.W.clone()]
for epoch in range(20):
    t_i = time.time()

    states = esn.train(X_train.float(), y_train.float())
    print("ESN trained")

    y_out = esn.predict(X_valid.float())
    y_pred = y_out
    report = src07_f.evaluate_regression_report(y_valid, y_pred)
    for m_name, m_val in report.items():
        if m_name not in hist:
            hist[m_name] = []
        hist[m_name].append(m_val)

    raster = (states > th).to(int)[:, -20:]
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
    new_W = ((new_W / new_W.max()) * 2 - 1) * esn.connections
    if epoch % 2 == 0:
        new_W = 0.5 * new_W + \
            torch.randn_like(esn.W) * 0.5 * esn.connections

    bad_diff = (
        (new_W !=0 ).to(int) - (reservoir_connections != 0).to(int)
        ).sum()
    assert bad_diff == 0
    
    esn.W = new_W
    assert esn.W.equal(new_W)
    W_hist.append(new_W)

    print(f"STDP executed in {time.time() - t_i:.0f}s")

print(f"t: {time.time() - t0:.0f}s")

print(f"STDP performance stats:")
perf_stats_after = pd.DataFrame(hist).describe()
print(perf_stats_after)

# %% STDP-Results: Explore weight changes

stdp_f.plot_most_changing_node_weights_and_connection(torch.stack(W_hist), n_top_weights=5)

# %% STDP-Results: Plot ESN connections

fig, ax = plt.subplots()
ax.set(
    title='Reservoir Connections'
)

stdp_f.connections_to_digraph(esn.connections, ax=ax)

# %%
