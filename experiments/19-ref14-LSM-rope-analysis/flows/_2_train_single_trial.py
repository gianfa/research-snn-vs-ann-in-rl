""" Topological search

For a given rope topology:
    RQ1. hat is the distribution of favourite positions vs (degree, radius)?


FC1 > LIF1 > FC2 > TanH

- input
    - feed reservoir subpopulation
    - I base input
- reservoir
    - sparse: rope (chain with skip connetions)
    - start mempot: random
    - Neuron
        - fixed beta
        - all excit
- readout
    - only a subpopulation is fed by I
    - returns a subpopulation to readout
    - computed on membrane pot
- loss
    - based on buffered y_pred and yi

- v Implement a queue to compute the accuracy. Use queue and torch.mode()
- v fix the function for the connections between neurons: mutual and 2 always!
- v increase neuronal connectivity
- v synchronize loss computation and optimization
- v dynamic synapses
- v add washout

Notes
- At first tune the minimum I may be beneficial.
    It may be done turning off the recurrent connections.
- Remember that Readout_LIF weights are free to move between -inf and inf
- Maybe a not-mutual chain architecture is more able to diffrentiate
- TUNING| a larger radius shows a more sustained activity. ~ reservoir/2
- TUNING| Adam, lr ~1e-2
- TUNING| increasing the `input_to_reservoir` from I, the acc increases
- TUNING| reservoir weight_dist "uniform", with approriate connection gain (which is now spread), the acc increases
- TUNING| reservoir weight_dist "normal" (.5, .4), boosted up
- TUNING| syn decay_rate very small (~.1)
- TUNING| loss_scope ~ 12 * subsig_len
- TUNING| normal distribution of decay rates seems to act as a stabilizer sometimes


sampling on same degre, radius, varying translation along k

References
----------
https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_5_FCN.ipynb
"""
# %%
import sys
import time

import seaborn as sns

sys.path += ['../', '../../../']
from _0_config import *

from experimentkit_in.logger_config import setup_logger

# %% Project Parameters

fd = 10 # forecast_delay

# RUN_PREFIX = f'report-500n-'
RUN_PREFIX = f'trash'

rename_with_params = True
exp_outdata_dir = EXP_DATA_DIR/'experiments'
data_path = EXP_DATA_DIR/"2freq_toy_ds-20000-sr_50-n_29.pkl"
description = "Performance Sampling vs radius and degree"


synapses_quantization_bins = None
plots = [
    'reservoir_adjacency',
    'raster',
    # 'graph'
]

# Project setup

run_name = RUN_PREFIX + "/" + ek.funx.generate_random_name()
RUN_OUTPUT_DIR = exp_outdata_dir/run_name

if not RUN_OUTPUT_DIR.exists():
    os.makedirs(RUN_OUTPUT_DIR)

logger = setup_logger(
    logger_name=RUN_PREFIX, log_file=RUN_OUTPUT_DIR/"log.log")

# Dataset loading
reload = False
if data_path.exists() and not reload:
    I_y = ek.funx.pickle_load(data_path)
    I, y = I_y['I'], I_y['y']
    data, targets = I, y
else:
    from _1_gen_input import I, y
    time.sleep(1.5)
    data, targets = I, y
    ek.funx.pickle_save_dict(data_path, {'I': I, 'y': y})

params = ek.funx.load_yaml(EXP_DIR/"flows/params.yaml")
ek.funx.yaml_save_dict(RUN_OUTPUT_DIR/"params.yaml", params)

# Data Preparation
#   data   [=] (1, n_samples)
#   targets [=] (n_samples, num_classes)

data = data.to(DEVICE).reshape(1, len(data))
targets = targets.to(DEVICE)
num_classes = 3
targets = torch.nn.functional.one_hot(targets.to(int), num_classes=num_classes)
# targets = y

# %%
# delta-encode
#   threshold must be smaller than the max value of data
# data = spikegen.delta(data.flatten().clone(), threshold=.1)


# %% train test split

if data.ndim == 1:
    data = data[None, :]
train_size = int(data.shape[1] * .75)
test_size = data.shape[1] - train_size

data_train = data[0, :train_size]
targets_train = targets[:train_size]
data_test = data[0, train_size:]
targets_test = targets[train_size:]

# %%
# Input definition

base_input_current_type = params['net']['base_input_current_type']
base_input_current_gain = params['net']['base_input_current_gain']
if base_input_current_type == 'constant':
    base_input = torch.ones_like(data) * base_input_current_gain
elif base_input_current_type == 'noisy':
    base_input = torch.rand_like(data) * base_input_current_gain
else:
    raise ValueError("unknown base_input_current_type")

#  --- Network Parameters ---
bs = 1

input_size = params['net']['input_size']
reservoir_size = params['net']['reservoir_size']
output_size = params['net']['output_size']

# selected subset input/output
reservoir_output_n = params['readout_LIF_connections']['reservoir_output_n']
reservoir_output = (
    torch.arange(reservoir_size).flip(dims=(0,))[:reservoir_output_n])

# Temporal Dynamics
num_steps = 1
beta = 0.95 #torch.normal(0.9, 0.1, size=(1, reservoir_size))  # 0.97 # 1.5
# beta = 0.95

# ---- Conn I-LIF ---

# the value of the non-zero LIF-I connections
conn_lif_I_gain = params['LIF_I_connections']['gain']
# n of reservoir neurons to feed by I
reservoir_fed_neurons_n = params['LIF_I_connections']['input_to_reservoir']

conn_lif_I = torch.zeros(input_size, reservoir_size)
reservoir_fed_neurons = \
    torch.arange(reservoir_fed_neurons_n)
    # torch.randperm(reservoir_size)[:reservoir_fed_neurons_n]
conn_lif_I[0, reservoir_fed_neurons] = conn_lif_I_gain

# ---- Conn LIF-LIF ---

enable_dyn_synapses = params['LIF_LIF_connections']['enable_dyn_synapses']
conn_lif_lif_gain = params['LIF_LIF_connections']['gain']

# conn_lif_lif = torch.normal(
# 0, 1, size=(reservoir_size, reservoir_size)) * conn_lif_lif_gain

# conn_lif_lif = src_19.topologies.gen_by_connection_degree(
#     reservoir_size, reservoir_size, degree=2)

# topology: positions
radius=params['LIF_LIF_connections']['radius']
degree=params['LIF_LIF_connections']['degree']
# topol_path = "/home/gianfrancesco/github/research-snn-vs-ann-in-rl/experiments/14-LSM-rope-analysis/data/experiments/trash/1700225607d4610558-CPse|alm_0.70-r_100-d_10/reservoir_topology.pkl"
# conn_lif_lif_topology = ek.funx.pickle_load(topol_path)['conn_lif_lif']

#  %%
conn_lif_lif_topology = src_19.topologies.gen_rope(
    reservoir_size, reservoir_size,
    radius=radius,
    degree=degree)

# topology: weights
conn_lif_lif_weigths_dist = params['LIF_LIF_connections']['weigths_dist']

if conn_lif_lif_weigths_dist == 'constant':
    conn_lif_lif_topology = conn_lif_lif_topology
if conn_lif_lif_weigths_dist == 'uniform':
    conn_lif_lif_topology *= torch.rand(conn_lif_lif_topology.shape)
if conn_lif_lif_weigths_dist == 'normal':
    conn_lif_lif_topology *= torch.normal(
        .5, .4, size=conn_lif_lif_topology.shape)

conn_lif_lif = conn_lif_lif_topology * conn_lif_lif_gain

# remove autapses
assert conn_lif_lif.diag().sum() == 0

# conn_lif_lif = torch.ones(reservoir_size, reservoir_size) * 0.05

# quantize
if synapses_quantization_bins is not None and synapses_quantization_bins>0:
    logger.info("Performing, quantization")
    conn_lif_lif = src_19.funx.quantize_to_bins(
        conn_lif_lif, synapses_quantization_bins)

ek.funx.pickle_save_dict(
    RUN_OUTPUT_DIR/"reservoir_topology.pkl",
    {'conn_lif_lif': conn_lif_lif})

lif1_has_autapsys = False
lif1_sparsity = 0
lif1_inhib_frac = 0


# Define the connection matrix between LIF and LIF
# This is intended like a static synapses adjacency matrix
# -inhibitors-
# inhib_idxs = torch.randperm(reservoir_size)[:int(reservoir_size * lif1_inhib_frac)]
# conn_lif_lif[inhib_idxs] = conn_lif_lif[inhib_idxs] * -1
# -autapsys-
if not lif1_has_autapsys:
    conn_lif_lif[torch.arange(reservoir_size), torch.arange(reservoir_size)] = 0
# -sparsity-
if lif1_sparsity > 0:
    lif1_dense_idxs = torch.argwhere(conn_lif_lif)
    lif1_tooff_idxs = lif1_dense_idxs[torch.randperm(
        len(lif1_dense_idxs))[:int(reservoir_size**2 * lif1_sparsity)]]
    conn_lif_lif[lif1_tooff_idxs[:, 0], lif1_tooff_idxs[:, 1]] = 0

#

decay_rate: torch.Tensor
syn_decay_rate = params['LIF_LIF_connections']['syn_decay_rate']
syn_decay_rate_dist = params['LIF_LIF_connections']['syn_decay_rate_dist']
if syn_decay_rate_dist == 'constant':
    decay_rate = syn_decay_rate
if syn_decay_rate_dist == 'uniform':
    decay_rate = torch.rand(size=(reservoir_size, 1)) * syn_decay_rate
if syn_decay_rate_dist == 'normal':
    decay_rate = torch.normal(
        0.5, 0.5, size=(reservoir_size, 1)).clip(0, 1) * syn_decay_rate

# --- LAYERS ---
fc1 = nn.Linear(input_size, reservoir_size)
lif1 = snn.Leaky(
    beta=beta,
    reset_mechanism="zero",
    learn_beta=params['net']['neuron_learn_beta'],
    learn_threshold=params['net']['neuron_learn_threshold'])
fc2 = nn.Linear(len(reservoir_output), output_size)
synapses = src_19.synapse.Synapse(
    size=(reservoir_size, reservoir_size),
    initial_conductance = params['LIF_LIF_connections'][
        'syn_initial_conductance'],
    decay_rate = decay_rate
)
spk_gain = 1

tanh = nn.Tanh()
sm = torch.nn.Softmax(dim=1)

# --- LOSS/OPTIM ---

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fc2.parameters(), lr=params['optim']['lr'])

# --- HIST ---

losses = []
mem_hist = {
    'filter': [torch.rand(bs, reservoir_size)],
    'readout': [torch.zeros(bs, output_size)]
}
spk_hist = {
    'filter': [torch.zeros(bs, reservoir_size)],
    'readout': [torch.zeros(bs, output_size)]
}

conn_readout_lif = []

acc_hist = []
cur_lif_lif_hist = []

#  --- TRAIN ---

epochs = params['training']['epochs']
loss_scope = params['training']['loss_scope']
eval_scope = loss_scope

buffer_capacity = params['training']['buffer_capacity']
yi_pred_buffer = src_19.funx.FIFO_buffer(capacity=buffer_capacity)
yi_pred_logits_buffer = src_19.funx.FIFO_buffer(capacity=buffer_capacity)
yi_buffer = src_19.funx.FIFO_buffer(capacity=buffer_capacity)
xi_yi_yi_pred_hist = []

washout_time = params['training']['washout_time']

# %%
for epoch in range(epochs):
    for i, xi_yi in enumerate(zip(data_train.t(), targets_train)):
        #if i == 1000: break

        xi, yi = xi_yi
        yi = yi.reshape(1, num_classes).to(DTYPE)
        for step in range(num_steps):
            spk_prev = spk_hist['filter'][-1]

            # synaptic update
            if enable_dyn_synapses:
                synapses.update(spk_prev)
                syn_contrib = spk_prev * synapses.synaptic_matrix * spk_gain
            else:
                syn_contrib = torch.ones(reservoir_size, reservoir_size)


            cur1_base =  (base_input[0, i] * torch.ones(1, reservoir_size))
            cur1_lif_I = (xi * conn_lif_I).reshape(bs, reservoir_size)
            # cur1_lif_lif [=] (reservoir_size,reservoir_size)
            cur1_lif_lif = conn_lif_lif * syn_contrib
            # cur1_lif_lif [=] (reservoir_size,reservoir_size)
            cur1_lif_lif = torch.matmul(spk_prev, conn_lif_lif)
            cur1 = cur1_base + cur1_lif_I + cur1_lif_lif

            spk1, mem1 = lif1(
                cur1,
                mem_hist['filter'][-1].squeeze().reshape(reservoir_size))
            
            selected_reservoir_output_activations = \
                mem1[:, reservoir_output]
            # cur2 = fc2(spk1)
            cur2 = fc2(selected_reservoir_output_activations)
            out = tanh(cur2).reshape(1, num_classes).to(DTYPE)
            
            yi_pred_logits = sm(out.reshape(1, 3))

            yi_pred_logits_buffer.push(yi_pred_logits)
            yi_pred_buffer.push(torch.argmax(yi_pred_logits).item())
            yi_buffer.push(yi)


            if i > 0 and i % loss_scope == 0 and i > washout_time:
                loss_val = loss_fn(
                    torch.stack(yi_pred_logits_buffer.buffer_[fd:]).squeeze(),
                    torch.stack(yi_buffer.buffer_[:-fd])
                        .squeeze().argmax(dim=1).to(torch.long))

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                losses.append(loss_val.clone().detach())

            # evaluate accuracy
            if i > eval_scope and i % eval_scope == 0 and i > washout_time:
                acc = (
                    torch.tensor(yi_pred_buffer.buffer_[fd:])
                        == torch.stack(yi_buffer.buffer_[:-fd])
                    .argmax(dim=2).squeeze()).mean(dtype=float)
                acc_hist.append(acc.item())

            # -- store hist --
            conn_readout_lif.append(
            {k: v.clone().detach() for k, v in fc2.named_parameters()})

            delay = 5
            mov_avg_yi_pred = (
                torch.tensor(yi_pred_buffer.buffer_[-delay:])
                    .float()
                    .mean()
                    .round())
            xi_yi_yi_pred_hist.append([
                xi, torch.argwhere(yi)[:, 1][0], mov_avg_yi_pred])
            cur_lif_lif_hist.append(cur1_lif_lif.clone().detach())
            mem_hist['filter'].append(mem1.clone().detach())
            spk_hist['filter'].append(spk1.clone().detach())
            mem_hist['readout'].append(out.clone().detach())


#
conn_readout_lif = {
    'weight': torch.stack([p['weight'] for p in conn_readout_lif]).detach(),
    'bias': torch.stack([p['bias'] for p in conn_readout_lif]).detach()
    }
losses = torch.stack(losses)
cur_lif_lif_hist = torch.stack(cur_lif_lif_hist)
mem_hist['filter'] = torch.stack(mem_hist['filter'])
mem_hist['readout'] = torch.stack(mem_hist['readout'])  # [=] (neuron_id, n_batches, bs, hidden_size)
spk_hist['filter'] = torch.stack(spk_hist['filter'])
spk_hist['readout'] = torch.stack(spk_hist['readout'])
xi_yi_yi_pred_hist = torch.tensor(xi_yi_yi_pred_hist)
acc_hist = torch.tensor(acc_hist)

fig, ax = plt.subplots()
ax.plot(losses)
ax.set_title('Loss')
ax.set_xlabel(f't/{loss_scope}')

fname = RUN_OUTPUT_DIR/"loss.png"
fig.savefig(fname)


fig, ax = plt.subplots()
ax.plot(conn_readout_lif['weight'][:, 0, 0])
ax.set_title('conn_readout_lif - 0,0')
fname = RUN_OUTPUT_DIR/"conn_readout_lif-0_0.png"
fig.savefig(fname)

fig, ax = plt.subplots()
ax.plot(acc_hist)
acc_mean = acc_hist.mean()
acc_last_mean = acc_hist[-int(len(acc_hist)*.1):].mean()
ax.set_title(f"Accuracy| mean: {acc_mean:.2f}, last 10% mean: {acc_last_mean:.2f}")
ax.set_xlabel("iteration/dw")
fname = RUN_OUTPUT_DIR/"accuracy.png"
fig.savefig(fname)

ek.funx.pickle_save_dict(
    RUN_OUTPUT_DIR/"results.pkl",
    {'acc_last_mean': acc_last_mean})

# %%

if rename_with_params is not None:
    new_name = f"{RUN_OUTPUT_DIR.name}|alm_{acc_last_mean:.2f}-r_{radius}-d_{degree}"
    RUN_OUTPUT_DIR.rename(RUN_OUTPUT_DIR.parent/new_name)
    RUN_OUTPUT_DIR = Path(RUN_OUTPUT_DIR.parent/new_name)
    assert RUN_OUTPUT_DIR.exists()

# %%

# Visualization

from matplotlib.ticker import MaxNLocator

n_neurons_to_plot = 5  # mem_hist['filter'].shape[2]
n_plots = (
    1 + mem_hist['readout'].shape[2] + n_neurons_to_plot)

fig = plt.figure(figsize=(7, 23))
plt.subplot(n_plots, 1, 1)
plt.plot(data.squeeze() * conn_lif_I[0, 0], c='k')
plt.title('$I$')

plot_start = 1 + 1
plot_end = plot_start + mem_hist['readout'].shape[2]
for i in range(plot_start, plot_end):
    ni = i - 2
    plt.subplot(n_plots, 1, i)
    plt.plot(mem_hist['readout'].squeeze()[:, ni], '.')
    plt.title(f"$Readout: V_{ni}$")
    plt.ylim(0, 1)
    # plt.xlim(0, 200)

plot_start = plot_start + mem_hist['readout'].shape[2]
plot_end = plot_start + n_neurons_to_plot
for i in range(plot_start, plot_end):
    ni = i - (2 + 3)# + 25
    plt.subplot(n_plots, 1, i)
    plt.plot(mem_hist['filter'].squeeze()[:, ni])
    plt.axhline(lif1.threshold.clone().detach(), linestyle='dotted', c='grey')
    plt.title(f"$Filter: V_{ni}$")
    # plt.xlim(2000, 2300)

fig.tight_layout()
fname = RUN_OUTPUT_DIR/"mempot-readout_and_reservoir.png"
fig.savefig(fname)


# LIF-LIF Connections Matrix, heatmap

if 'reservoir_adjacency' in plots:
    fig, ax = plt.subplots(figsize=(10, 10))
    if synapses_quantization_bins is not None:
        ek.visualization.plot_heatmap_discrete(
            conn_lif_lif, conn_lif_lif.unique().tolist())
    else:
        sns.heatmap(
            conn_lif_lif, cmap='binary', cbar=True,
            linewidths=0, linecolor='grey', ax=ax)
    ax.set_xlabel('Neuron id')
    ax.set_ylabel('Neuron id')
    ax.set_title('LIF-LIF Connections Matrix')
    fname = RUN_OUTPUT_DIR/"reservoir-conn_matrix.png"
    plt.show()
    fig.savefig(fname)

#
if 'raster' in plots:
    ax = src_19.visualization.generate_raster_plot(spk_hist['filter'].squeeze().T)
    ax.set_xlim(0, 2000)
    fname = RUN_OUTPUT_DIR/"reservoir-raster.png"
    ax.figure.savefig(fname)

# %%

if 'graph' in plots:
    fig, ax = plt.subplots()
    ek.funx.connections_to_digraph(conn_lif_lif, ax=ax, axis_on=True)
    fname = RUN_OUTPUT_DIR/"topol_graph.png"
    fig.savefig(fname)

# %% Evaluation



losses_test = []
acc_hist_test = []

yi_pred_buffer_test = src_19.funx.FIFO_buffer(capacity=buffer_capacity)
yi_pred_logits_buffer_test = src_19.funx.FIFO_buffer(capacity=buffer_capacity)
yi_buffer_test = src_19.funx.FIFO_buffer(capacity=buffer_capacity)
xi_yi_yi_pred_hist_test = []


mem_hist_test = {
    'filter': [torch.rand(bs, reservoir_size)],
    'readout': [torch.zeros(bs, output_size)]
}

for i, xi_yi in enumerate(zip(data_test.t(), targets_test)):
    xi, yi = xi_yi
    yi = yi.reshape(1, num_classes).to(DTYPE)
    for step in range(num_steps):

        # synaptic update
        if enable_dyn_synapses:
            synapses.update(spk_prev)
            syn_contrib = spk_prev * synapses.synaptic_matrix * spk_gain
        else:
            syn_contrib = torch.ones(reservoir_size, reservoir_size)


        cur1_base =  (base_input[0, i] * torch.ones(1, reservoir_size))
        cur1_lif_I = (xi * conn_lif_I).reshape(bs, reservoir_size)
        # cur1_lif_lif [=] (reservoir_size,reservoir_size)
        cur1_lif_lif = conn_lif_lif * syn_contrib
        # cur1_lif_lif [=] (reservoir_size,reservoir_size)
        cur1_lif_lif = torch.matmul(spk_prev, conn_lif_lif)
        cur1 = cur1_base + cur1_lif_I + cur1_lif_lif

        spk1, mem1 = lif1(
            cur1,
            mem_hist_test['filter'][-1].squeeze().reshape(reservoir_size))
        
        selected_reservoir_output_activations = \
            mem1[:, reservoir_output]
        # cur2 = fc2(spk1)
        cur2 = fc2(selected_reservoir_output_activations)
        out = tanh(cur2).reshape(1, num_classes).to(DTYPE)
        
        yi_pred_logits = sm(out.reshape(1, 3))

        yi_pred_logits_buffer_test.push(yi_pred_logits)
        yi_pred_buffer_test.push(torch.argmax(yi_pred_logits).item())
        yi_buffer_test.push(yi)


        if i > 0 and i % loss_scope == 0 and i > washout_time:
            loss_val = loss_fn(
                torch.stack(yi_pred_logits_buffer_test.buffer_).squeeze(),
                torch.stack(yi_buffer_test.buffer_)
                    .squeeze().argmax(dim=1).to(torch.long))
            losses_test.append(loss_val.clone().detach())

        # evaluate accuracy
        if i > loss_scope and i > washout_time:
            acc = (
                torch.tensor(yi_pred_buffer_test.buffer_)
                    == torch.stack(yi_buffer_test.buffer_)
                .argmax(dim=2).squeeze()).mean(dtype=float)
            acc_hist_test.append(acc.item())


        delay = 5
        mov_avg_yi_pred = (
            torch.tensor(yi_pred_buffer_test.buffer_[-delay:])
                .float()
                .mean()
                .round())
        xi_yi_yi_pred_hist_test.append([
            xi, torch.argwhere(yi)[:, 1][0], mov_avg_yi_pred])

        mem_hist_test['filter'].append(mem1.clone().detach())

# %%

losses_test = torch.stack(losses_test)

xi_yi_yi_pred_hist_test = torch.tensor(xi_yi_yi_pred_hist_test)
acc_hist_test = torch.tensor(acc_hist_test)
mem_hist_test['filter'] = torch.stack(mem_hist_test['filter'])

# %%

fig, ax = plt.subplots()
ax.plot(losses_test)
ax.set_title('Loss')
ax.set_xlabel(f't/{loss_scope}')

fname = RUN_OUTPUT_DIR/"loss_test.png"
fig.savefig(fname)
# %%

fig, ax = plt.subplots()
ax.plot(acc_hist_test)
acc_mean = acc_hist_test.mean()
ax.set_title(f"Accuracy| mean: {acc_mean:.2f}")
ax.set_xlabel("iteration/dw")
fname = RUN_OUTPUT_DIR/"accuracy_test.png"
fig.savefig(fname)


# %%

print(f"'{RUN_OUTPUT_DIR}' done")



# %%

# xs = xi_yi_yi_pred_hist.shape[0]

# for i in range(3):
#     cat_i_idx = torch.argwhere(xi_yi_yi_pred_hist[:, 1] == i)
#     plt.plot(cat_i_idx, xi_yi_yi_pred_hist[cat_i_idx, 0], label=i)
# plt.legend()

#%%

# diffrc = abs(xi_yi_yi_pred_hist[:, 2] - xi_yi_yi_pred_hist[:, 1])
# # plt.plot(diffrc)
# plt.plot(diffrc.view(10, -1).mean(dim=0))
# plt.axhline(0.5, color='grey')
# plt.title("$|y_pred - y_i|$")