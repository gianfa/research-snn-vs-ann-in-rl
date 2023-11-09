"""


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
        ax=None,
        **args) -> plt.Axes:
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


# %% -- Project Parameters --

trials = [
    'trial-r4-d2',  # r4, d2
    'trial-r4-d3',  # r4, d3
    'trial-r4-d1',  # r4, d1
    'trial-r4-d4',  # r4, d4
    'trial-r3-d1',  # r3, d1
    'trial-r3-d2',  # r3, d2
    'trial-r3-d3',  # r3, d3
    'trial-r2-d2',  # r2, d2
    'trial-r2-d1',  # r2, d1
    'trial-r1-d1',  # r1, d1
]

OUTPUT_DIR = EXP_REPORT_DIR/"topological_analysis"
expected_run_n = 100

if not OUTPUT_DIR.exists():
    os.mkdir(OUTPUT_DIR)


# %% -- Analyze --


fig_pos_perf, axs_pos_perf = plt.subplots(4, 4)
fig_pos_perf_top, axs_pos_perf_top = plt.subplots(4, 4)
fig_top_graph, axs_top_graph = plt.subplots(4, 4)

for i, RUN_PREFIX in enumerate(trials):
    # data_path = EXP_DATA_DIR/"2freq_toy_ds-20000-sr_50-n_29.pkl"

    RUN_DIR = EXP_DATA_DIR/'experiments'/RUN_PREFIX
    subrun_paths = [run_i for run_i in RUN_DIR.iterdir() if run_i.is_dir()]
    assert len(subrun_paths) == expected_run_n, \
        f"{len(subrun_paths)} runs found for '{RUN_PREFIX}'," \
        + f"expected {expected_run_n}"

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
    ax.figure.savefig(fname)

    pos_perf_filtered = torch.where(
        pos_perf.mean(0) > .25, pos_perf.mean(0), 0)


    ax = plot_adjacency_map(
        pos_perf_filtered,
        title=f'Mean final acc per position > 0.25\nr:{radius}, d:{degree}')
    fname = OUTPUT_DIR/f"avg_acc_per_pos-ressize_20-d_{degree}-r_{radius}.png"
    ax.figure.savefig(fname)

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
fig_pos_perf.savefig(OUTPUT_DIR/f"perf_x_pos-mean.png")
fig_pos_perf

fig_pos_perf_top.suptitle("Top connections by mean-performance")
fig_pos_perf_top.tight_layout()
fig_pos_perf_top.savefig(OUTPUT_DIR/f"top_perf_x_pos-mean.png")
fig_pos_perf_top

fig_top_graph.suptitle("Top connections by mean-performance, graph")
fig_top_graph.tight_layout()
fig_top_graph.savefig(OUTPUT_DIR/f"top_perf_graph.png")
fig_top_graph

# %%

# trial_prefix = 'S2'
# for trial_i in EXP_REPORT_DIR.iterdir():
#     if trial_i.name.startswith(trial_prefix):







# %%
