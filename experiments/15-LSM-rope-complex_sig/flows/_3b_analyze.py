"""


After researching the most important hyperparameters,
we chose a configuration to analyze.
reservoir_size: 20, radius: 4, degree [2, 3].


"""
# %%
from math import ceil
from pathlib import Path
import sys

sys.path += ['../', '../../../']

import string
import networkx as nx

from _0_config import *
from experimentkit_in.logger_config import setup_logger

# %%

SAVE_DATA = False

# %% -- Helpers --

def plot_acc_hist(acc_hist, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(acc_hist)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("acc scope, after wash time")
    return ax


def plot_adjacency_map(
        adj: torch.tensor,
        colour_diagonal: float = None,
        ax=None, **args) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    fig = ax.figure

    adj_ = adj.clone()
    if colour_diagonal:
        for ri in range(pos_perf_mean.shape[0]):
            adj_[ri, ri] = colour_diagonal

    sns.heatmap(
    adj_, vmin=0, vmax=1, linewidths=0.2,
    ax=ax, cmap='coolwarm').set(**args)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return ax


def connections_to_digraph(
    connections: torch.Tensor,
    colour_groups: dict = {},
    show: bool = True,
    relabel = False,
    axis_on = False,
    ax: plt.Axes = None
) -> nx.MultiDiGraph:
    # TODO: fix position of the neurons
    G = nx.from_numpy_matrix(
        connections.numpy(),
        parallel_edges=True,
        create_using=nx.MultiDiGraph()
    )

    if relabel:
        label_mapping = {
            i: string.ascii_lowercase[i] for i in range(connections.shape[0])}
        G = nx.relabel_nodes(G, label_mapping)
    G.edges(data=True)

    # colour the nodes
    colors = {}
    for color, nodes in colour_groups.items():
        for node in nodes:
            colors[node] = color

    if show:
        if ax is None:
            _, ax = plt.subplots(1, 1)
        pos = nx.spring_layout(G)
        nx.draw(
            G, pos, with_labels=True, ax=ax,
            node_color=[colors.get(node, 'blue') for node in G.nodes()],
            font_color='white',
            font_size=8,
            edgecolors='black',
            node_size=30)
        if axis_on:
            ax.axis('on')
        return G, ax
    return G


def moving_average(signal, window_size):
    cumsum = np.cumsum(signal)
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def plot_signal_band_plot(signal, smooth=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    q1 = torch.quantile(signal, 0.25, dim=0)
    q2 = torch.quantile(signal, 0.50, dim=0)
    q3 = torch.quantile(signal, 0.75, dim=0)

    if smooth:
        w_size = 5
        q1 = moving_average(q1, w_size)
        q2 = moving_average(q2, w_size)
        q3 = moving_average(q3, w_size)

    ax.plot(range(len(q1)), q1, label="Pct25", color='black', linewidth=.7)
    ax.plot(range(len(q2)), q2, label="Pct50", color='red', linewidth=.7)
    ax.plot(range(len(q3)), q3, label="Pct75", color='black', linewidth=.7)
    ax.fill_between(range(len(q1)), q1, q3, color='gray', alpha=0.3)
    return ax


# %% -- Project Parameters --


TRIALS_NAME_1 = '15-trial-NO-plastic'
TRIALS_NAME_2 = '15-trial-YES-plastic'

OUTPUT_DIR = EXP_REPORT_DIR/"topological_analysis"
expected_run_n = 100

if not OUTPUT_DIR.exists():
    os.mkdir(OUTPUT_DIR)


# %%
n_samples = 100

trials_NO_dir = Path(EXP_DATA_DIR/f"experiments/{TRIALS_NAME_1}")
assert trials_NO_dir.is_dir()

subtrial_NO_dirs = list(trials_NO_dir.iterdir())[:n_samples]

print(f"subtrial_NO_dirs: {len(subtrial_NO_dirs)}")

trials_YES_dir = Path(EXP_DATA_DIR/f"experiments/{TRIALS_NAME_2}")
assert trials_YES_dir.is_dir()

subtrial_YES_dirs = list(trials_YES_dir.iterdir())[:n_samples]

print(f"subtrial_YES_dirs: {len(subtrial_YES_dirs)}")

# %% Load results

trials_NO_results = [
    ek.funx.pickle_load(s/"results.pkl") for s in subtrial_NO_dirs]
trials_YES_results = [
    ek.funx.pickle_load(s/"results.pkl") for s in subtrial_YES_dirs]

# %%

trials_NO_results_acc_last_mean = [
    x['acc_last_mean'] for x in trials_NO_results]
trials_NO_results_acc_hist_list = [x['acc_hist'] for x in trials_NO_results]


trials_YES_results_acc_last_mean = [
    x['acc_last_mean'] for x in trials_YES_results]
trials_YES_results_acc_hist_list = [x['acc_hist'] for x in trials_YES_results]

# %% Plot: hist of last acc_mean

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

ax_no = plot_acc_hist(trials_NO_results_acc_hist_list[3], ax=axs[0])
ax_no.set_title("No plasticity")
ax_yes = plot_acc_hist(trials_YES_results_acc_hist_list[3], ax=axs[1])
ax_yes.set_title("With plasticity")
fig.savefig(EXP_REPORT_DIR/"compare-examples-plasticity_acc_hist-plot.png")

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].hist(trials_NO_results_acc_last_mean, bins=20)
axs[0].set_title("No plasticity")
axs[0].axvline(np.array(trials_YES_results_acc_last_mean).mean(), color="grey")
axs[1].hist(trials_YES_results_acc_last_mean, bins=20)
axs[1].set_title("With plasticity")
axs[1].axvline(np.array(trials_YES_results_acc_last_mean).mean(), color="grey")
fig.savefig(EXP_REPORT_DIR/"compare-examples-plasticity_last_acc_mean-hist.png")

# %% Plot band of the acc

from scipy.ndimage import gaussian_filter1d

fig, axs = plt.subplots(1, 2, sharey=True)

# smooth=True
trials_YES_results_acc_hist = torch.stack(trials_YES_results_acc_hist_list)

plot_signal_band_plot(trials_YES_results_acc_hist, ax=axs[0])
axs[0].set_ylabel("Accuracy")
axs[0].set_xlabel("acc scope, after wash time")
axs[0].set_title("With plasticity")

trials_NO_results_acc_hist = torch.stack(trials_NO_results_acc_hist_list)

plot_signal_band_plot(trials_NO_results_acc_hist, ax=axs[1])
axs[1].set_ylabel("Accuracy")
axs[1].set_xlabel("acc scope, after wash time")
axs[1].set_title("Without plasticity")

fig.tight_layout()

# %%

SAVE_DATA and fig_pos_perf.savefig(OUTPUT_DIR/f"compare-perf_x_pos-mean.png")

# %%

start_from = 10_000
fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

sig_len = trials_NO_results_acc_hist[:, start_from:].shape[1]
plot_signal_band_plot(trials_NO_results_acc_hist[:, start_from:], ax=axs[1])
axs[0].set_ylabel("Accuracy")
axs[0].set_xlabel("acc scope, after wash time")
axs[0].set_title("Without plasticity")
axs[0].set_xlim(0, sig_len)

plot_signal_band_plot(trials_YES_results_acc_hist[:, start_from:], ax=axs[0])
axs[1].set_ylabel("Accuracy")
axs[1].set_xlabel("acc scope, after wash time")
axs[1].set_title("With plasticity")

fig.suptitle(f"Average Accuracy. Last {sig_len} windows")

fig.tight_layout()

# %%

fig, ax = plt.subplots()

plot_signal_band_plot(trials_NO_results_acc_hist[:, start_from:], ax=ax)
plot_signal_band_plot(trials_YES_results_acc_hist[:, start_from:], ax=ax)
ax.set_ylabel("Accuracy")
ax.set_xlabel("acc scope, after wash time")
ax.set_title("Without plasticity")
ax.set_xlim(0, sig_len)
ax.legend(["No plasticty", "With plasticity"])

# %%

exit()

# %% -- Analyze --


fig_pos_perf, axs_pos_perf = plt.subplots(4, 4)
fig_pos_perf_top, axs_pos_perf_top = plt.subplots(4, 4)
fig_top_graph, axs_top_graph = plt.subplots(4, 4)

for i, RUN_PREFIX in enumerate(trials):
    # data_path = EXP_DATA_DIR/"2freq_toy_ds-20000-sr_50-n_29.pkl"

    RUN_DIR = EXP_DATA_DIR/'experiments'/RUN_PREFIX
    subrun_paths = [run_i for run_i in RUN_DIR.iterdir() if run_i.is_dir()]
    assert (len(subrun_paths) == expected_run_n,
        f"{len(subrun_paths)} runs found, expected {expected_run_n}")

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

    pos_perf_mean = pos_perf.mean(0)

    ax = plot_adjacency_map(
        pos_perf_mean,
        title=f'r:{radius}, d:{degree}',
        ax = axs_pos_perf[radius-1, degree-1])
    fname = OUTPUT_DIR/f"avg_acc_per_pos-ressize_20-d_{degree}-r_{radius}.png"
    SAVE_DATA and ax.figure.savefig(fname)

    pos_perf_filtered = torch.where(
        pos_perf.mean(0) > .25, pos_perf.mean(0), 0)


    ax = plot_adjacency_map(
        pos_perf_filtered,
        title=f'Mean final acc per position > 0.25\nr:{radius}, d:{degree}')
    fname = OUTPUT_DIR/f"avg_acc_per_pos-ressize_20-d_{degree}-r_{radius}.png"
    SAVE_DATA and ax.figure.savefig(fname)

    # Top neurons per row
    TOP_N = degree

    top_neurons = torch.zeros_like(pos_perf_mean)
    for ri in range(pos_perf_mean.shape[0]):
        top_idxs = (torch.argsort(pos_perf_mean[ri], descending=True)[:TOP_N])
        top_neurons[ri][top_idxs] = 1

    plot_adjacency_map(
        top_neurons, colour_diagonal=0.5,
        title=f'Top {TOP_N} per row', ax=axs_pos_perf_top[radius-1, degree-1])


    connections_to_digraph(
        top_neurons, ax=axs_top_graph[radius-1, degree-1], axis_on=True)

    #

    print(topols.shape)
    plot_adjacency_map(topols.mean(0)).set(title='Mean weight values')

axs_to_turn_off = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
for ax_i, ax_j in axs_to_turn_off:
    axs_pos_perf[ax_i, ax_j].axis('off')
    axs_pos_perf_top[ax_i, ax_j].axis('off')
    axs_top_graph[ax_i, ax_j].axis('off')

fig_pos_perf.suptitle("Mean-performance per neuron-neuron connection")
fig_pos_perf.tight_layout()
SAVE_DATA and fig_pos_perf.savefig(OUTPUT_DIR/f"perf_x_pos-mean.png")
fig_pos_perf

fig_pos_perf_top.suptitle("Top connections by mean-performance")
fig_pos_perf_top.tight_layout()
SAVE_DATA and fig_pos_perf_top.savefig(OUTPUT_DIR/f"top_perf_x_pos-mean.png")
fig_pos_perf_top

fig_top_graph.suptitle("Top connections by mean-performance, graph")
fig_top_graph.tight_layout()
SAVE_DATA and fig_top_graph.savefig(OUTPUT_DIR/f"top_perf_graph.png")
fig_top_graph

# %%

# trial_prefix = 'S2'
# for trial_i in EXP_REPORT_DIR.iterdir():
#     if trial_i.name.startswith(trial_prefix):


