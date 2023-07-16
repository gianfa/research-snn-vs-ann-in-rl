""" Lorenz Forecast ESN + STDP: Performance vs SignalShift


Exp Observations
----------------


References
----------
1. SCHAETTI, Nils; SALOMON, Michel; COUTURIER, Raphaël. Echo state networks-based reservoir computing for mnist handwritten digits recognition. In: 2016 IEEE Intl Conference on Computational Science and Engineering (CSE) and IEEE Intl Conference on Embedded and Ubiquitous Computing (EUC) and 15th Intl Symposium on Distributed Computing and Applications for Business Engineering (DCABES). IEEE, 2016. p. 484-491.

"""
# %%
from copy import deepcopy
import itertools
import os
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

import src08.funx as src08_f

ROOT = Path('../../../')
DATA_DIR = ROOT/'data'
EXP_DIR = ROOT/'experiments/08-ESN-STDP-MLops'
EXP_DATA_DIR = EXP_DIR/'data'
EXP_REPORT_DIR = EXP_DIR/'report'

SUBEXP_DIR = EXP_DATA_DIR/'exp-2'
SUBEXP_RESULTS_DIR = SUBEXP_DIR/'produced_data'

assert EXP_DIR.exists(), \
f"CWD: {Path.cwd()}\nROOT: {ROOT.absolute()}"

recompute = False


# %% Training

# params = {
# 'data': {
#     'example_len': 10000,
#     'test_size': 0.2,
#     'valid_size': 0.15,
#     'shift': 10
# },
# 'experiment': {
#     'n_trials': 50,
#     'n_STDP_steps': 5,
# },
# 'model': {
#     'reservoir_size': 100,
#     'STDP_scope': 20,
# }
# }

exp_id = 'exp_len_shift_STDPsteps_scope'

grid_params_names = [
    'shift', 'example_len',
    'n_STDP_steps', 'STDP_scope',
    'reservoir_size', 'decay']

print(f"{len(list((SUBEXP_RESULTS_DIR).iterdir()))} Trials found")

recompute = False

# %% Plot Performance comparison before/after. boxplots

def perf_register_get_or_create(perf_register_fpath):
    if perf_register_fpath.exists():
        return pickle_load(perf_register_fpath)['perfs']
    else:
        return []

perf_fpath = SUBEXP_DIR/'produced_data_cache/perf.pkl'
if perf_fpath.exists():
    perf = pd.read_pickle(perf_fpath)
else:
    perf = pd.DataFrame()

perf_register_path = SUBEXP_DIR/'produced_data_cache/perf_register.pkl'
perf_register = perf_register_get_or_create(perf_register_path)



all_exp = list(SUBEXP_RESULTS_DIR.iterdir())
all_exp_names = [ei.name for ei in all_exp]
new_exps = list(set(all_exp_names).difference(set()))
new_exps_paths = [SUBEXP_RESULTS_DIR/ei_name for ei_name in new_exps]

print(f"{len(new_exps_paths)} new experiments")

# %%


for i, exp_dir in enumerate(new_exps_paths):
    try:
        exp_params = pickle_load(exp_dir/"params.pkl")

        perf_hist_nonstdp = pd.DataFrame(
            pickle_load(exp_dir/'perf_hist_nonstdp.pkl'))
        perf_hist_nonstdp['example_len'] = exp_params['data']['example_len']
        perf_hist_nonstdp['shift'] = exp_params['data']['shift']
        perf_hist_nonstdp['n_STDP_steps'] = exp_params[
            'experiment']['n_STDP_steps'] 
        perf_hist_nonstdp['STDP_scope'] = exp_params['experiment']['STDP_scope'] 
        perf_hist_nonstdp['reservoir_size'] = exp_params['model']['reservoir_size']
        perf_hist_nonstdp['decay'] = exp_params['model']['decay']
        perf_hist_nonstdp['type'] = 'before'

        perf_hist_after_stdp = pd.DataFrame(
            pickle_load(exp_dir/'perf_hist_after_stdp.pkl'))

        perf_hist_after_stdp['example_len'] = exp_params['data']['example_len']
        perf_hist_after_stdp['shift'] = exp_params['data']['shift']
        perf_hist_after_stdp['n_STDP_steps'] = exp_params[
            'experiment']['n_STDP_steps'] 
        perf_hist_after_stdp['STDP_scope'] = exp_params['experiment']['STDP_scope']  
        perf_hist_after_stdp['reservoir_size'] = exp_params['model']['reservoir_size']
        perf_hist_after_stdp['decay'] = exp_params['model']['decay']
        perf_hist_after_stdp['type'] = 'after'

        if len(perf) == 0:
            perf = pd.concat((perf_hist_nonstdp, perf_hist_after_stdp), axis=0)
            # cache perf
        else:
            perf = pd.concat(
                (perf, perf_hist_nonstdp, perf_hist_after_stdp), axis=0)
        perf.to_pickle(perf_fpath)
        
        # register new experiment added to perf
        with open(perf_register_path, '+a') as f:
            f.write(exp_dir.name)

        del perf_hist_nonstdp, perf_hist_after_stdp
    except Exception as e:
        print(f"{e}")

print(f"{i} Trials found")

# %% boxplot before vs after, removing outliers

for m_name in ['r2', 'mse']:
    for param_name in grid_params_names:
        fig, ax = plt.subplots()
        sns.boxplot(
        data=perf,
        x=param_name, y=m_name, hue='type',
        showfliers=False, ax=ax) #, showmeans=True)

        # fig.savefig(
        #     SUBEXP_DIR/f"results/comparison-{m_name}_vs_{param_name}.png")

# %% stratify by shift

for m_name in ['r2', 'mse']:
    for param_name in grid_params_names:
        sns.relplot(
            data=perf,
            x=param_name, y=m_name, hue='type',
            col='shift', row='reservoir_size',
            ax=ax) #, showmeans=True)

# %% Plot Performance comparison before/after. dot-whiskers

diffs = pd.DataFrame()
for i, exp_dir in enumerate((SUBEXP_RESULTS_DIR).iterdir()):

    exp_params = pickle_load(exp_dir/"params.pkl")

    perf_hist_nonstdp = pd.DataFrame(
        pickle_load(exp_dir/'perf_hist_nonstdp.pkl'))
    perf_hist_after_stdp = pd.DataFrame(
        pickle_load(exp_dir/'perf_hist_after_stdp.pkl'))    

    diff = (perf_hist_after_stdp['r2'] - perf_hist_nonstdp['r2']).to_frame()
    diff['mse'] = perf_hist_after_stdp['mse'] - perf_hist_nonstdp['mse']
    diff['mae'] = perf_hist_after_stdp['mae'] - perf_hist_nonstdp['mae']

    diff['example_len'] = exp_params['data']['example_len']
    diff['shift'] = exp_params['data']['shift']
    diff['n_STDP_steps'] = exp_params['experiment']['n_STDP_steps'] 
    diff['STDP_scope'] = exp_params['experiment']['STDP_scope'] 

    del perf_hist_nonstdp, perf_hist_after_stdp

    if len(diffs) == 0:
        diffs = diff
    else:
        diffs = pd.concat(
            (diffs, diff), axis=0)

    metric = 'r2'
    param = 'shift'
    mean = diffs.sort_values(param).groupby(param)[metric].mean().to_list()
    std = diffs.sort_values(param).groupby(param)[metric].std().to_list()

    fig, ax = plt.subplots()
    ax.errorbar(
    diffs[param].sort_values().unique(),
    mean, yerr=std, fmt='o', capsize=5)
    ax.plot(diffs[param].sort_values().unique(), mean, linewidth=0.4)

    ax.set_xlabel(f'{param}')
    ax.set_ylabel(f'{metric} mean')
    ax.set_title(r'')
    # ax.set_ylim(0, 10)


    plt.show()

# %% Plot Performance comparison before/after: means. scatter

diffs = []
for i, exp_dir in enumerate((SUBEXP_RESULTS_DIR).iterdir()):
    diff = {}
    try:
        exp_params = pickle_load(exp_dir/"params.pkl")
        perf_hist_nonstdp = pd.DataFrame(
            pickle_load(exp_dir/'perf_hist_nonstdp.pkl'))
        perf_hist_after_stdp = pd.DataFrame(
            pickle_load(exp_dir/'perf_hist_after_stdp.pkl')) 

        for metric in ['r2', 'mse', 'mae']:
            diff_s = (perf_hist_after_stdp[metric] - perf_hist_nonstdp[metric])
            diff[f'{metric}_mean'] = diff_s.mean()
            diff[f'{metric}_std'] = diff_s.std()
        
        diff['example_len'] = exp_params['data']['example_len']
        diff['shift'] = exp_params['data']['shift']
        diff['n_STDP_steps'] = exp_params['experiment']['n_STDP_steps'] 
        diff['STDP_scope'] = exp_params['experiment']['STDP_scope']
        diffs.append(diff)
    except Exception as e:
        print(i)
    
diffs = pd.DataFrame(diffs)


metric = 'mae'
param = 'shift'

mean = diffs[f"{metric}_mean"].to_list()
std = diffs[f"{metric}_std"].to_list()

# %% Plot Performance comparison before/after: delta metrics

for m_name in ['r2', 'mse']:
    for param_name in grid_params_names:
        try:
            diffs = diffs.sort_values(param_name)

            fig, ax = plt.subplots()
            ax.scatter(
                diffs[param_name].unique(),
                diffs.groupby(param_name)[f"{m_name}_mean"].mean(),
                s=0.8)
            # ax.errorbar(
            #     diffs[param],
            #     mean, yerr=std, fmt='.', capsize=5)
            # ax.plot(diffs[param].sort_values().unique(), mean, linewidth=0.4)

            ax.plot(
            diffs[param_name].unique(),
            diffs.groupby(param_name)[f"{m_name}_mean"].mean(), linewidth=0.6)
            ax.set_xlabel(f'{param_name}')
            ax.set_ylabel(f'$\Delta${m_name} mean')
            ax.set_title(r'')
            # ax.set_ylim(0, 10)
        except Exception as e:
            print(f"{e}")

# %%



# %%

metric = 'r2'
param = 'shift'
mean = diffs.sort_values(param).groupby(param)[metric].mean().to_list()
std = diffs.sort_values(param).groupby(param)[metric].std().to_list()

fig, ax = plt.subplots()
ax.errorbar(
diffs[param].sort_values().unique(),
mean, yerr=std, fmt='o', capsize=5)
ax.plot(diffs[param].sort_values().unique(), mean, linewidth=0.4)

ax.set_xlabel(f'{param}')
ax.set_ylabel(f'{metric} mean')
ax.set_title(r'')
ax.set_ylim(0, 10)

plt.show()

# %% Load Weight history

W_hist_stdp = pickle_load(exp_dir/'W_hist_stdp.pkl')['W_hist_stdp']
W_hist_nonstdp = pickle_load(exp_dir/'W_hist_nonstdp.pkl')['W_hist_nonstdp']

print(
f"W_hist_stdp.shape: {W_hist_stdp.shape}"
+ f"\nW_hist_nonstdp.shape: {W_hist_nonstdp.shape}")

metrics_to_plot=['mse', 'mae', 'r2']

# %% Plot Performance comparison

W_hist_stdp_orig = deepcopy(W_hist_stdp)

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
