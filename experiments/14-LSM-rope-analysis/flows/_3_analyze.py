"""


After researching the most important hyperparameters,
we chose a configuration to analyze.
reservoir_size: 20, radius: 4, degree [2, 3].



Done
----
(radius, degreee)
4,1
4,2
4,3


TODO
----
(radius, degreee)
4,4
3,1
3,2
3,3
3,1
3,2
3,3
2,2
2,1
1,1

"""
# %%
import sys

sys.path += ['../', '../../../']
from _0_config import *

from experimentkit_in.logger_config import setup_logger

# %% Project Parameters

prefixes = [
    'S2-trial1',
    'S2-trial2',
    'S2-trial3',   # 
    # 'S2-trial4', # d4, r4
]

OUTPUT_DIR = EXP_REPORT_DIR/"topological_analysis"

if not OUTPUT_DIR.exists():
    os.mkdir(OUTPUT_DIR)

for RUN_PREFIX in prefixes:
    # data_path = EXP_DATA_DIR/"2freq_toy_ds-20000-sr_50-n_29.pkl"

    RUN_DIR = EXP_REPORT_DIR/RUN_PREFIX
    subrun_paths = [run_i for run_i in RUN_DIR.iterdir() if run_i.is_dir()]

    exp_params = ek.funx.load_yaml(subrun_paths[0]/"params.yaml")
    degree = exp_params['LIF_LIF_connections']['degree']
    radius = exp_params['LIF_LIF_connections']['radius']

    def collect_from_runs(x):
        return [run_i/x for run_i in subrun_paths]

    topol_paths = collect_from_runs('reservoir_topology.pkl')
    topols = torch.stack([ek.funx.pickle_load(topol_path)['conn_lif_lif']
            for topol_path in topol_paths])

    acc_lastm_paths = collect_from_runs('results.pkl')
    acc_lastms = torch.stack(
        [ek.funx.pickle_load(acc_lastm_path)['acc_last_mean']
            for acc_lastm_path in acc_lastm_paths])


    pos_perf = (acc_lastms.view(len(acc_lastms), 1, 1) * (topols!=0))
    """ Performance of the positions
    Each topology has the performance value in each of the cells"""

    fig, ax = plt.subplots()
    sns.heatmap(pos_perf.mean(0), vmin=0, vmax=1, cmap='coolwarm').set(
        title=f'Mean performance per position\nr:{radius}, d:{degree}')
    fname = OUTPUT_DIR/f"avg_acc_per_pos-ressize_20-d_{degree}-r_{radius}.png"
    fig.savefig(fname)

    pos_perf_filtered = torch.where(
        pos_perf.mean(0) > .25, pos_perf.mean(0), 0)

    fig, ax = plt.subplots()
    sns.heatmap(pos_perf_filtered, vmin=0, vmax=1, cmap='coolwarm', ax=ax).set(
        title=f'Mean final acc per position > 0.25\nr:{radius}, d:{degree}')
    fname = OUTPUT_DIR/f"avg_acc_per_pos-ressize_20-d_{degree}-r_{radius}-25pc.png"
    fig.savefig(fname)


    # %%

    print(topols.shape)
    sns.heatmap(topols.mean(0)).set(title='Mean weight values', cmap='coolwarm')

# %%

# trial_prefix = 'S2'
# for trial_i in EXP_REPORT_DIR.iterdir():
#     if trial_i.name.startswith(trial_prefix):



