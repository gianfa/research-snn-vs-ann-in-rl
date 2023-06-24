""" Lorenz Forecast ESN + STDP: Performance vs SignalShift

Q: Is it possible to achieve an improvement in the classification
    performance of an ESN by applying STDP?


-------------------------

Exp Observations
----------------
1. It is prone to exploding weights since STDP will amplify indefinitely.
    -> Normalization before update.
2. 

   
References
----------
1. SCHAETTI, Nils; SALOMON, Michel; COUTURIER, Raphaël. Echo state networks-based reservoir computing for mnist handwritten digits recognition. In: 2016 IEEE Intl Conference on Computational Science and Engineering (CSE) and IEEE Intl Conference on Embedded and Ubiquitous Computing (EUC) and 15th Intl Symposium on Distributed Computing and Applications for Business Engineering (DCABES). IEEE, 2016. p. 484-491.

TODOs
-----
* v differenza stdp non stdp 
in funzione della lunghezza del segnale
* distribuzione dei pesi nei confronti delle performance


"""
# %%
from copy import deepcopy
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List

from dvclive import Live
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch

sys.path += ["..", "../..", "../../.."]
from experimentkit_in.visualization import get_cmap_colors
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

recompute = False


# %% Training

params = {
    'data': {
        'example_len': 10000,
        'test_size': 0.2,
        'valid_size': 0.15,
        'shift': 10
    },
    'experiment': {
        'n_trials': 50,
        'n_STDP_steps': 5,
    },
    'model': {
        'reservoir_size': 100,
    }
}


reservoir_size = params['model']['reservoir_size']

# %%
with Live(save_dvc_exp=True) as live:

    # STDP-Execute
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
    live.log_param("trials", params['experiment']['n_trials'])

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

    n_trials = params['experiment']['n_trials']
    n_STDP_steps = params['experiment']['n_STDP_steps']
    verbose = False

    # Data Loading
    X_train, X_valid, X_test, y_train, y_valid, y_test = \
        src07_f.expt_generate_new_lorenz_data(
            example_len = params['data']['example_len'],
            test_size = params['data']['test_size'],
            valid_size = params['data']['valid_size'],
            recompute = True,
            ds_path = EXP_DATA_DIR/f"ds_lorenz.pkl",
            shift = params['data']['shift'],  # forecasted delay
            s=12, r=30, b=2.700
        )

    W_hist_nonstdp = []
    W_hist_stdp =  []

    for i in range(n_trials):
    
        print(f"INFO: Trial #{i}")

        esn = BaseESN(
            input_size = X_train.shape[1],
            reservoir_size=reservoir_size,
            output_size = y_train.shape[1],
            connections = (stdp_f.generate_simple_circle_connections_mask(
                (reservoir_size, reservoir_size)) > 0).int(),
        )

        # Non-STDP
        state_hist = esn.train(X_train.float(), y_train.float())
        y_out = esn.predict(X_valid.float())
        y_pred = y_out
        
        report = src07_f.evaluate_regression_report(y_valid, y_pred)
        for m_name, m_val in report.items():
            if m_name not in perf_hist_nonstdp:
                perf_hist_nonstdp[m_name] = []
            perf_hist_nonstdp[m_name].append(m_val)
            live.log_metric(m_name, m_val)
        
        W_hist_nonstdp.append(esn.W.data)
        
        # STDP Update
        perf_hist_stdp_inner = {}
        W_hist_stdp.append([])
        for epoch in range(n_STDP_steps):
            t_i = time.time()

            # STDP
            raster = (state_hist > th).to(int)[:, -20:]
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

            if verbose: print(f"STDP executed in {time.time() - t_i:.0f}s")

            # # Retrain after STDP and evaluate
            state_hist = esn.train(X_train.float(), y_train.float())
            y_out = esn.predict(X_valid.float())
            y_pred = y_out
            
            report = src07_f.evaluate_regression_report(y_valid, y_pred)
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

        print(f"t: {time.time() - t0:.0f}s")

    print(f"STDP performance stats:")
    perf_stats_before = pd.DataFrame(perf_hist_nonstdp).describe()
    print(perf_stats_before)


    perf_stats_after = pd.DataFrame(perf_hist_after_stdp).describe()
    print(perf_stats_after)

    live.next_step()

# %% Plot Performance comparison before/after. boxplots


metrics_to_plot=['mse', 'mae', 'r2']
fig, axs = plt.subplots(1, len(metrics_to_plot), figsize=(10, 8))
for ax, mi in zip(axs.ravel(), metrics_to_plot):
    src07_f.plot_compare_df_via_boxplot(
        df1=perf_stats_before[[mi]],
        df2=perf_stats_after[[mi]],
        names = ['before', 'after'],
        xlabel='',
        title='',
        ax=ax
    )
ax.figure.tight_layout()
ax.figure.suptitle("Performance comparison before and after STDP", y=1.05)


# %%

W_hist_stdp_orig = deepcopy(W_hist_stdp)

W_hist_stdp = [torch.stack(W_hist_stdp_i) for W_hist_stdp_i in W_hist_stdp]
W_hist_stdp = torch.stack(W_hist_stdp)  # (trials, stdp_steps, W_h, W_w)

# eigenvalues history
ev_hist = []
for trial_i in W_hist_stdp:
    ev_hist.append(
        torch.stack([torch.linalg.eigvals(W_step) for W_step in trial_i]))

ev_hist = torch.stack(ev_hist)  # (trials, stdp_steps, 1)


# %%    Plot Evolution of eigenvalues along steps in many trials

fig, axs = plt.subplots(len(ev_hist), figsize=(4, 35))

for i, ax in enumerate(axs.ravel()):
    src07_f.plot_eigenvalues_tensor(ev_hist[i], ax=ax)
    title = f"Eig. trial #{i}.\n"
    title += "\n".join([
        f"MSE: {perf_hist_after_stdp['mse'][i]:.2f}",
        f"MAE: {perf_hist_after_stdp['mae'][i]:.2f}",
        f"R2: {perf_hist_after_stdp['r2'][i]:.2f}",
        f"Spec.Radius: {ev_hist[i].abs().max():.2f}"
        ])
    ax.set_title(title, fontsize=7)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()



# %%    Plot Evolution of Max eigenvalues along steps in many trials

fig, axs = plt.subplots(2,1)
sr_x_step, _ = torch.max(ev_hist.abs(), dim=2)  # spectral radius
mse_all = torch.stack([perf_i['mse'] for perf_i in perf_hist_stdp])


sr_x_step = pd.DataFrame(
    sr_x_step, columns=[f'step #{i+1}' for i in range(5)])
sns.boxplot(data=sr_x_step, orient='h', ax=axs[0])

# plot means
for i,ci in enumerate(sr_x_step.columns):
    ax.plot(
        sr_x_step[ci].mean(), i, marker='o',
        color='black')

axs[0].set_title("Spec.Radius distribution per STDP steps along trials")
fig.tight_layout()


# %% Plot Performance comparison: metrics diff

# plt.plot(perf_hist_nonstdp['mae'], label='non-stdp')
# plt.plot(perf_hist_after_stdp['mae'], label='stdp')
diff = np.array(
    perf_hist_after_stdp['mse']) - np.array(perf_hist_nonstdp['mse'])
plt.plot(diff, label='$MSE_{STDP} - MSE_{non-STDP}$')
plt.legend()
plt.grid()
plt.show()


diff = np.array(
    perf_hist_after_stdp['mae']) - np.array(perf_hist_nonstdp['mae'])
plt.plot(diff, label='$MAE_{STDP} - MAE_{non-STDP}$')
plt.legend()
plt.grid()
plt.show()

diff = np.array(perf_hist_after_stdp['r2']) - np.array(perf_hist_nonstdp['r2'])
plt.plot(diff, label='performance diff: R2')
plt.legend()
plt.grid()
plt.show()

# %% Plot STDP weights evo: heatmap
# NB. heatmap NEEDS the ylabel, otherwise doesn't work
#     -> issue to rise?
import matplotlib.pyplot as plt
import torch

# take one trial (vs all STDP steps)
stacked_weights = W_hist_stdp[0]

# flatten: dims = (STDP_step, all_connections)
stacked_weights = W_hist_stdp[0].view(n_STDP_steps, -1)

fig, ax = plt.subplots()
sns.heatmap(
    stacked_weights.T,
    center=0,
    cmap='bwr', ax=ax)
ax.set_ylabel('STDP step')
ax.set_xlabel('connection Id')

# %%

# Single step weights, non-flatten
fig, ax = plt.subplots()
sns.heatmap( W_hist_stdp[0, 0], center=0, cmap='bwr', ax=ax)
ax.set_title('STDP step n.1 of the trial n.1')

# %% Plot STDP weights evo: scatter

fig, ax = plt.subplots()
for ri in range(stacked_weights.shape[0]):
    ax.scatter(
        range(stacked_weights.shape[1]),
        stacked_weights[ri, :],
        s=0.2, label=f"step {ri}")
ax.legend()

# %% weights mean vs R2 per STDP step

# last_W_hist_stdp_avg = W_hist_stdp[-1].mean(axis=1).mean(axis=1).shape
# last_r2_hist = perf_hist_after_stdp['r2']

# W_vs_r2 = np.array(
#     [(x, r2) for x, r2 in zip(last_W_hist_stdp_avg, last_r2_hist)])
# fig, ax = plt.subplots()
# ax.scatter(
#     np.arange(W_vs_r2.shape[0]), W_vs_r2[:, 0], c=W_vs_r2[:, 1]/abs(W_vs_r2[:, 1]).max(), marker='o', cmap='coolwarm')
# ax.set(title='W mean', xlabel="# step")

# cbar = fig.colorbar()
# cbar.set_label('R2')


# %% STDP-Results: Explore weight changes

axs = stdp_f.plot_most_changing_node_weights_and_connection(
    W_hist_stdp[0], n_top_weights=5)

# %% STDP-Results: Plot ESN connections

fig, ax = plt.subplots()
ax.set(
    title='Reservoir Connections'
)

stdp_f.connections_to_digraph(esn.connections, ax=ax)

# %% ---------------------------------------
# %% Plot all eveolution in a trial, inspect

# %% Plot Summary:

# intra trial: STDP steps
fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=True)

stdp_f.plot_most_changing_node_weights(
    W_hist_stdp[0], n_top_weights=5, ax=axs[0])

# axs[0].set_xlim(0, n_STDP_steps)
axs[0].set_xlabel('STDP step')
axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

sns.heatmap(
    stacked_weights.T,
    center=0,
    cmap='bwr', ax=axs[1])

# add vertical grid
for i in range(stacked_weights.shape[0]):
    axs[1].axvline(i, linewidth=0.5, color='grey')

ax.set_xlabel('STDP step')
ax.set_ylabel('connection Id')

# %% Plot Summary:

# inter trial: trials
fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

diff = np.array(
    perf_hist_after_stdp['mse']) - np.array(perf_hist_nonstdp['mse'])
axs[1].plot(range(n_trials), diff)
axs[1].set_title('$MSE_{STDP} - MSE_{non-STDP}$')
axs[1].grid()


diff = np.array(
    perf_hist_after_stdp['mae']) - np.array(perf_hist_nonstdp['mae'])
axs[2].plot(range(n_trials), diff)
axs[2].set_title('$MAE_{STDP} - MAE_{non-STDP}$')
axs[2].grid()

diff = np.array(perf_hist_after_stdp['r2']) - np.array(perf_hist_nonstdp['r2'])
axs[3].plot(range(n_trials), diff)
axs[3].set_title('$R^2_{STDP} - R^2_{non-STDP}$')
axs[3].grid()

sr_x_step = pd.DataFrame(
    sr_x_step, columns=[f'step #{i+1}' for i in range(5)])
sns.boxplot(data=sr_x_step, orient='v', ax=axs[0])
axs[0].set_title("Spec.Radius distribution per STDP steps along trials")

# # plot means
# for i,ci in enumerate(sr_x_step.columns):
#     ax.plot(
#         sr_x_step[ci].mean(), i, marker='o',
#         color='black')

# %%
