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
import itertools
import os
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List

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
from experimentkit_in.reporting import ReportMD
from stdp.estimators import BaseESN
from stdp import funx as stdp_f

# import src07.funx as src07_f

# %%
ROOT = Path('/home/gianfrancesco/github/research-snn-vs-ann-in-rl')
DATA_DIR = ROOT/'data'
EXP_DIR = ROOT/'experiments/07-ESN-STDP'
EXP_DATA_DIR = EXP_DIR/'data'
EXP_REPORT_DIR = EXP_DIR/'report'

# SUBEXP_DIR = EXP_DATA_DIR/'exp-example_len-shift-n_STDP_steps-STDP_scope'
EXP_NAME = 'exp-5'
SUBEXP_DIR = EXP_DATA_DIR/EXP_NAME
SUBEXP_PROD_DATA = SUBEXP_DIR/'produced_data'
SUBEXP_RESULTS_DIR = SUBEXP_DIR/'results'

FROM_WANDB = True

assert ROOT.resolve().name == 'research-snn-vs-ann-in-rl'
assert EXP_DIR.exists(), \
f"CWD: {Path.cwd()}\nROOT: {ROOT.absolute()}"
# %%

recompute = False
report_path = EXP_REPORT_DIR/(str(EXP_NAME) + ".md")

reporter = ReportMD(report_path)

# clear the report
reporter.clean_all_the_text(imsure=True)
reporter = ReportMD(report_path, f"# {EXP_NAME}")



# %% Training
grid_params_names = ['shift', 'example_len', 'n_STDP_steps', 'STDP_scope']

# %% Load Performance data

perf = pd.DataFrame()  # := (n_trial, scores)
delta_perf = pd.DataFrame()  # := (scores)
for i, exp_dir in enumerate((SUBEXP_PROD_DATA).iterdir()):
    try:
        exp_params = pickle_load(exp_dir/"params.pkl")
        if not FROM_WANDB: exp_params = exp_params['data']

        # get the trials performance := (n_trial, scores)
        perf_hist_nonstdp = pd.DataFrame(
            pickle_load(exp_dir/'perf_hist_nonstdp.pkl'))
        for k, v in exp_params.items():
            perf_hist_nonstdp[k] = exp_params[k]
        perf_hist_nonstdp['type'] = 'before'
        perf_hist_nonstdp['trial'] = i

        perf_hist_after_stdp = pd.DataFrame(
            pickle_load(exp_dir/'perf_hist_after_stdp.pkl'))
        for k, v in exp_params.items():
            perf_hist_after_stdp[k] = exp_params[k]
        perf_hist_after_stdp['type'] = 'after'
        perf_hist_after_stdp['trial'] = i

        if len(perf) == 0:
            perf = pd.concat((perf_hist_nonstdp, perf_hist_after_stdp), axis=0)
        else:
            perf = pd.concat(
                (perf, perf_hist_nonstdp, perf_hist_after_stdp), axis=0)

        # Delta
        # we choose to summarize through median because is robust to outliers
        temp_delta = pd.DataFrame({
            'mse_before': [perf_hist_nonstdp['mse'].median()],
            'mae_before': [perf_hist_nonstdp['mae'].median()],
            'mse_after': [perf_hist_after_stdp['mse'].median()],
            'mae_after': [perf_hist_after_stdp['mae'].median()],
            'shift': perf_hist_after_stdp.loc[0, 'shift'],
            'example_len': perf_hist_after_stdp.loc[0, 'example_len'],
            'reservoir_size': perf_hist_after_stdp.loc[0, 'reservoir_size'],
            'n_STDP_steps': perf_hist_after_stdp.loc[0, 'n_STDP_steps'],
            'STDP_scope': perf_hist_after_stdp.loc[0, 'STDP_scope'],
            'n_trials': perf_hist_after_stdp.loc[0, 'n_trials']
        })
        temp_delta['delta_mse'] = temp_delta['mse_after'] \
            - temp_delta['mse_before']

        delta_perf = pd.concat((delta_perf, temp_delta))

        del perf_hist_nonstdp, perf_hist_after_stdp, temp_delta
    except Exception as e:
        print(f"{exp_dir}: {e}")

# pickle_save_dict(
#     SUBEXP_RESULTS_DIR/f"comparison-perf.pkl",
#     {'perf': perf})

# pickle_save_dict(
#     SUBEXP_RESULTS_DIR/f"comparison-delta_errors.pkl",
#     {'delta_perf': delta_perf})


# %% boxplot before vs after, removing outliers

for m_name in ['r2', 'mse']:
    for param_name in grid_params_names:
        fig, ax = plt.subplots()
        sns.boxplot(
        data=perf, x=param_name, y=m_name, hue='type',
        showfliers=False, ax=ax) #, showmeans=True)

        img_path = SUBEXP_RESULTS_DIR/f"comparison-{m_name}_vs_{param_name}.png"
        fig.savefig(img_path)
        img_tag_md = f'<img src="{img_path}">'
        # src07_f.report_md_append(f"\n\n\n\n{img_tag_md}", report_path)

# %% FaceGrid, boxplot before vs after, removing outliers

compby = "shift"
g = sns.FacetGrid(perf, col=compby, col_wrap=3, sharey=False)
g.map_dataframe(
    sns.boxplot, x="STDP_scope", y="mse", hue="type", showfliers=False)
g.add_legend()
fig = plt.gcf()

img_path = SUBEXP_RESULTS_DIR/f"comparison-stratif-mse_vs_STDP_scope_by_{compby}.png"
fig.savefig(img_path)
img_tag_md = f'<img src="{img_path}">'
# src07_f.report_md_append(f"\n\n{img_tag_md}", report_path)


compby = "example_len"
g = sns.FacetGrid(perf, col=compby, col_wrap=3, sharey=False)
g.map_dataframe(
    sns.boxplot, x="STDP_scope", y="mse", hue="type", showfliers=False)
g.add_legend()
fig = plt.gcf()
img_path=SUBEXP_RESULTS_DIR/f"comparison-stratif-mse_vs_STDP_scope_by_{compby}.png"
fig.savefig(img_path)
img_tag_md = f'<img src="{img_path}">'
# src07_f.report_md_append(f"\n\n{img_tag_md}", report_path)

compby = "n_STDP_steps"
g = sns.FacetGrid(perf, col=compby, col_wrap=3, sharey=False)
g.map_dataframe(
    sns.boxplot, x="STDP_scope", y="mse", hue="type", showfliers=False)
g.add_legend()
fig = plt.gcf()
img_path = SUBEXP_RESULTS_DIR/f"comparison-stratif-mse_vs_STDP_scope_by_{compby}.png"
fig.savefig(img_path)
img_tag_md = f'<img src="{img_path}">'
# src07_f.report_md_append(f"\n\n{img_tag_md}", report_path)

# %% FaceGrid, boxplot before vs after, mse vs shift by STDP_scope = 10

perf_STDP_scope_10 = perf[perf.STDP_scope==10]
sns.boxplot(perf_STDP_scope_10, x='shift', y='mse', hue="type", showfliers=False)

# partitions per STDP_scope 10, shift 9
perf_STDP_scope_10[perf_STDP_scope_10['shift']==9].groupby('type').size()

# %%  

delta_perf = pickle_load(
    SUBEXP_RESULTS_DIR/f"comparison-delta_errors.pkl")['delta_perf']

fig, ax = plt.subplots()
sns.histplot(delta_perf['mse_after']/delta_perf['mse_before'], fill=False, ax=ax)
ax.axvline(1, color='grey')
ax.set_title('$MSE_{after}/MSE_{before}$')
img_path = SUBEXP_RESULTS_DIR/f"distrib_delta_mse-hist.png"
fig.savefig(img_path)

# %% difference per each run

perf = pickle_load(
    SUBEXP_RESULTS_DIR/f"comparison-perf.pkl")['perf']

perf = perf.reset_index()

print("all differences distribution")
delta_single_run = pd.Series(perf[perf['type']=='after']['mse'].values - perf[perf['type']=='before']['mse'].values)
delta_single_run.hist(bins=50).set(title="all diffs")
plt.show()
print(delta_single_run.describe())


print("\nall differences > 0 distribution")
delta_single_run[delta_single_run>0].hist().set(title='diffs > 0')
plt.show()
print(delta_single_run[delta_single_run>0].describe())


print("\nall 0< differences < 10 distribution")
delta_single_run[(delta_single_run>0) & (delta_single_run<10)].hist(bins=50,).set(title='0 < diffs < 10')
plt.show()
print(delta_single_run[(delta_single_run>0) & (delta_single_run<10)].describe())

# all together
zoomed_statistics = pd.concat((
    delta_single_run.describe(),
    delta_single_run[delta_single_run>0].describe(),
    delta_single_run[(delta_single_run>0) & (delta_single_run<10)].describe()
), axis=1)
zoomed_statistics.columns = ['all diffs', 'diffs > 0', '0 < diffs < 10']
zoomed_statistics


# %% ----- Significance Test -------
# Mann-Whitney U test on SHAP values

def stats_compute_difference_ci_via_bs(
        A: np.array,
        B: np.array,
        ci: int,
        num_bootstrap_samples: int =1000
    ):

    bootstrap_mean_diffs = np.zeros(num_bootstrap_samples)

    # Bootstrap sampling and sampled differences computation
    for i in range(num_bootstrap_samples):
        bootstrap_A = np.random.choice(A, size=len(A), replace=True)
        bootstrap_B = np.random.choice(B, size=len(B), replace=True)
        bootstrap_mean_diffs[i] = np.mean(bootstrap_A) - np.mean(bootstrap_B)

    ci_upper = ci + (100 - ci)/2
    ci_lower = (100 - ci)/2
    assert ci_upper - ci_lower == ci
    return np.percentile(
        bootstrap_mean_diffs, [ci_lower, ci_upper])

import numpy as np
from scipy.stats import mannwhitneyu

U_results = {}

feature_name = 'mse'
A = perf[perf['type']=='after'].groupby('trial').first()['mse'].values
B = perf[perf['type']=='before'].groupby('trial').first()['mse'].values

# A = perf[perf['type']=='after']['mse'].values
# B = perf[perf['type']=='before']['mse'].values
U_statistic, p_value = mannwhitneyu(A, B)
ci_lower, ci_upper = stats_compute_difference_ci_via_bs(A, B, 95)

U_results[feature_name] = {
    'p': p_value,
    'U': U_statistic,
    'M.size': len(A),
    'F.size': len(B),
    'M.median': np.median(A),
    'F.median': np.median(B),
    'diff ci_low(95%)': ci_lower,
    'diff ci_upper(95%)': ci_upper
}

significance_th = 0.05
if p_value >= significance_th:
    U_results[feature_name][f'p<{significance_th}'] = False
    print("There are no significant differences between the two datasets")
else:
    U_results[feature_name][f'p<{significance_th}'] = True
    print(f"U statistic: {U_statistic}", U_statistic)
    print(f"p value: {p_value:.5f}")
    print("There are significant differences between the two datasets")
    print(f"medians:\n\tA:{np.median(A)}\n\tB:{np.median(B)}")
    print(f"size:\n\tA:{np.median(A)}\n\tB:{np.median(B)}")

U_results = pd.DataFrame(U_results)

# %%  Plot delta of single runs
fig, ax = plt.subplots()
sns.histplot(delta_single_run, fill=False, ax=ax)
ax.axvline(1, color='grey')
ax.set_title('$MSE_{after}/MSE_{before}$')


delta_improvement_idx = pd.Series(np.argwhere((delta_single_run > 0) & (delta_single_run < 1)).flatten())

#TODO
# dataframe_image.export(
#     delta_single_run.describe(),
#     str(SUBEXP_RESULTS_DIR/"comparison-delta_mse-describe-df.png"))

# %% Plot Performance comparison before/after. dot-whiskers

diffs = pd.DataFrame()
for i, exp_dir in enumerate((SUBEXP_PROD_DATA).iterdir()):

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
for i, exp_dir in enumerate((SUBEXP_PROD_DATA).iterdir()):
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
        fig.savefig(
            SUBEXP_RESULTS_DIR/f"comparison-delta_{m_name}_vs_{param_name}.png")



# %%

metric = 'mse'
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

fig, axs = plt.subplots(len(ev_hist), figsize=(6, 45))

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
