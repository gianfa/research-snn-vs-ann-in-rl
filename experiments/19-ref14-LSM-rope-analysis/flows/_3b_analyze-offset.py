""" TOREVIEW! maybe is not analyze_dataset


After researching the most important hyperparameters,
we chose a configuration to analyze.
reservoir_size: 20, radius: 4, degree [2, 3].





"""
# %%
from math import ceil
import sys

sys.path += ['../', '../../../']

import string
import networkx as nx

from _0_config import *
from experimentkit_in.logger_config import setup_logger

# %% -- Helpers --

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


def collect_from_runs(x):
    return [run_i/x for run_i in subrun_paths]

# %% -- Project Parameters --

# TRIAL_NAME = '14-topol-trial1-s50-r16_-d2-quant3'
TRIAL_NAME = '14-topol-trial4-s50-r16_-d2-quant3'
OUTPUT_PREFIX = '14_1_s50_quant_3'

OUTPUT_DIR = EXP_REPORT_DIR/"topological_analysis"
expected_run_n = 100

if not OUTPUT_DIR.exists():
    os.mkdir(OUTPUT_DIR)


# %% -- Analyze --

# Set Paths
RUN_PREFIX = TRIAL_NAME
run_dir = EXP_DATA_DIR/'experiments'/RUN_PREFIX
subrun_paths = [run_i for run_i in run_dir.iterdir() if run_i.is_dir()]
assert (len(subrun_paths) == expected_run_n,
    f"{len(subrun_paths)} runs found, expected {expected_run_n}")

# Get Params
exp_params = ek.funx.load_yaml(subrun_paths[0]/"params.yaml")
degree = exp_params['LIF_LIF_connections']['degree']
radius = exp_params['LIF_LIF_connections']['radius']



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
    title=f'r:{radius}, d:{degree}')
fname = OUTPUT_DIR/f"{OUTPUT_PREFIX}-avg_acc_per_pos-ressize_50-d_{degree}-r_{radius}.png"
ax.figure.savefig(fname)

pos_perf_filtered = torch.where(
    pos_perf.mean(0) > .25, pos_perf.mean(0), 0)


ax = plot_adjacency_map(
    pos_perf_filtered,
    title=f'Mean final acc per position > 0.25\nr:{radius}, d:{degree}')
fname = OUTPUT_DIR/f"{OUTPUT_PREFIX}-avg_acc_per_pos-ressize_50-d_{degree}-r_{radius}.png"
ax.figure.savefig(fname)

# Top neurons per row
TOP_N = degree

top_neurons = torch.zeros_like(pos_perf_mean)
for ri in range(pos_perf_mean.shape[0]):
    top_idxs = (torch.argsort(pos_perf_mean[ri], descending=True)[:TOP_N])
    top_neurons[ri][top_idxs] = 1

ax = plot_adjacency_map(
    top_neurons, colour_diagonal=0.5,
    title=f'Top {TOP_N} connections by mean performance')
ax.figure.savefig(OUTPUT_DIR/f"{OUTPUT_PREFIX}-top_perf_graph.png")

fig, ax = plt.subplots()
connections_to_digraph(top_neurons, axis_on=True, ax=ax)
ax.set_title("Top {TOP_N} connections graph")
fig.savefig(OUTPUT_DIR/f"{OUTPUT_PREFIX}-top_perf_graph.png")

#

print(topols.shape)
plot_adjacency_map(topols.mean(0)).set(title='Mean weight values')


# %%

# trial_prefix = 'S2'
# for trial_i in EXP_REPORT_DIR.iterdir():
#     if trial_i.name.startswith(trial_prefix):


