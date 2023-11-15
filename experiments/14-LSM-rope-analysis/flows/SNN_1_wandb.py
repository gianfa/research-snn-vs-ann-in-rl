""" With noise injection

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

References
----------
https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_5_FCN.ipynb
"""
# %%
import sys

import wandb

sys.path += ['../', '../../../']
from _0_config import *

# from src_14 import funx, synapse, topologies, visualization
# from experimentkit_in.logger_config import setup_logger
import matplotlib
matplotlib.use('agg')

wandb.login()

# %% Project Parameters


# RUN_PREFIX = f'14-trial8-rx-dx-quant-3'
RUN_PREFIX = f'14-topol-trial9-s50-r16_-d2-quant3-delta_enc'
data_path = EXP_DATA_DIR/"2freq_toy_ds-20000-sr_50-n_29.pkl"
synapses_quantization_bins = 3

# %% Project setup

run_name = RUN_PREFIX + "/" + ek.funx.generate_random_name()
RUN_REPORT_DIR = EXP_REPORT_DIR/run_name

if not RUN_REPORT_DIR.exists():
    os.makedirs(RUN_REPORT_DIR)

logger = setup_logger(
    logger_name=RUN_PREFIX, log_file=RUN_REPORT_DIR/"log.log")

# %% Dataset loading

reload = True
if data_path.exists() and reload:
    I_y = ek.funx.pickle_load(data_path)
    I, y = I_y['I'], I_y['y']
    data, targets = I, y
else:
    from _1_gen_input import I, y
    time.sleep(1.5)
    data, targets = I, y
    ek.funx.pickle_save_dict(data_path, {'I': I, 'y': y})



params = ek.funx.load_yaml(EXP_DIR/"flows/sweep_config.yaml")
ek.funx.yaml_save_dict(RUN_REPORT_DIR/"params.pkl", params)

# %% Data Preparation

data = data.to(DEVICE).reshape(1, len(data))
targets = targets.to(DEVICE)
num_classes = 3
targets = torch.nn.functional.one_hot(targets.to(int), num_classes=num_classes)
# targets = y
# %%
# # ENCODE #
#data = data.clip(min=0)
# delta-encode
data = spikegen.delta(data.clone(), threshold=.1)

# %%
def wandb_main():
    wandb.init(project=f'{EXP_DIR.name}|{RUN_PREFIX}', entity='gianfa')
    config = wandb.config
    # %% Input definition

    base_input_current_type = config['base_input_current_type']
    base_input_current_gain = config['base_input_current_gain']
    if base_input_current_type == 'constant':
        base_input = torch.ones_like(data) * base_input_current_gain
    elif base_input_current_type == 'noisy':
        base_input = torch.rand_like(data) * base_input_current_gain
    else:
        raise ValueError("unknown base_input_current_type")

    # %%  --- Network Parameters ---
    bs = 1

    input_size = 1
    reservoir_size = config['net_reservoir_size']
    output_size = 3

    # selected subset input/output
    reservoir_output_n = int(config['ro_lif_output_n'])
    reservoir_output = (
        torch.arange(reservoir_size).flip(dims=(0,))[:reservoir_output_n])

    # Temporal Dynamics
    num_steps = 1
    beta = 0.95 #torch.normal(0.9, 0.1, size=(1, reservoir_size))  # 0.97 # 1.5
    # beta = 0.95

    # ---- Conn I-LIF ---

    # the value of the non-zero LIF-I connections
    conn_lif_I_gain = config['lif_i_gain']
    # n of reservoir neurons to feed by I
    reservoir_fed_neurons_n = int(config['lif_i_input_n'])

    conn_lif_I = torch.zeros(input_size, reservoir_size)
    reservoir_fed_neurons = \
        torch.arange(reservoir_fed_neurons_n)
        # torch.randperm(reservoir_size)[:reservoir_fed_neurons_n]
    conn_lif_I[0, reservoir_fed_neurons] = conn_lif_I_gain

    # %% ---- Conn LIF-LIF ---

    enable_dyn_synapses = config['enable_dyn_synapses']
    conn_lif_lif_gain = config['lif_lif_gain']

    # conn_lif_lif = torch.normal(
    # 0, 1, size=(reservoir_size, reservoir_size)) * conn_lif_lif_gain

    # conn_lif_lif = src_14.topologies.gen_by_connection_degree(
    #     reservoir_size, reservoir_size, degree=2)

    # topology: positions
    conn_lif_lif_topology = src_14.topologies.gen_rope(
        reservoir_size, reservoir_size,
        radius=int(config['radius']),
        degree=int(config['degree']))

    # topology: weights
    conn_lif_lif_weigths_dist = config['lif_lif_weights_dist']

    if conn_lif_lif_weigths_dist == 'constant':
        conn_lif_lif_topology = conn_lif_lif_topology
    if conn_lif_lif_weigths_dist == 'uniform':
        conn_lif_lif_topology *= torch.rand(conn_lif_lif_topology.shape)
    if conn_lif_lif_weigths_dist == 'normal':
        conn_lif_lif_topology *= torch.normal(
            .5, .4, size=conn_lif_lif_topology.shape)

    conn_lif_lif = conn_lif_lif_topology * conn_lif_lif_gain

    assert conn_lif_lif.diag().sum() == 0

    # conn_lif_lif = torch.ones(reservoir_size, reservoir_size) * 0.05

    # quantize
    if synapses_quantization_bins is not None and synapses_quantization_bins>0:
        logger.info("Performing, quantization")
        conn_lif_lif = src_14.funx.quantize_to_bins(
            conn_lif_lif, synapses_quantization_bins)

    ek.funx.pickle_save_dict(
        RUN_REPORT_DIR/"reservoir_topology.pkl",
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

    # %%

    decay_rate: torch.Tensor
    syn_decay_rate = config['syn_decay_rate']
    syn_decay_rate_dist = config['syn_decay_rate_dist']
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
        learn_beta=config['neuron_learn_beta'],
        learn_threshold=config['neuron_learn_threshold'])
    fc2 = nn.Linear(len(reservoir_output), output_size)
    synapses = src_14.synapse.Synapse(
        size=(reservoir_size, reservoir_size),
        initial_conductance = config[
            'syn_initial_conductance'],
        decay_rate = decay_rate
    )
    spk_gain = 1

    tanh = nn.Tanh()
    sm = torch.nn.Softmax(dim=1)

    # --- LOSS/OPTIM ---

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fc2.parameters(), lr=config['optim_lr'])

    # %% --- HIST ---

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

    # %%  --- TRAIN ---

    epochs = config['training_epochs']
    loss_scope = config['training_loss_scope']
    eval_scope = loss_scope

    buffer_capacity = config['training_buffer_capacity']
    yi_pred_buffer = ek.collections_utils.FIFO_buffer(capacity=buffer_capacity)
    yi_pred_logits_buffer = ek.collections_utils.FIFO_buffer(capacity=buffer_capacity)
    yi_buffer = ek.collections_utils.FIFO_buffer(capacity=buffer_capacity)

    washout_time = config['training_washout_time']

    for epoch in range(epochs):
        for i, xi_yi in enumerate(zip(data.t(), targets)):
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
                        torch.stack(yi_pred_logits_buffer.buffer_).squeeze(),
                        torch.stack(yi_buffer.buffer_)
                            .squeeze().argmax(dim=1).to(torch.long))
                    wandb.log({"loss_val": loss_val})

                    optimizer.zero_grad()
                    loss_val.backward()
                    optimizer.step()

                    losses.append(loss_val.clone().detach())

                # evaluate accuracy
                if i > loss_scope and i > washout_time:
                    acc = (
                        torch.tensor(yi_pred_buffer.buffer_)
                            == torch.stack(yi_buffer.buffer_)
                        .argmax(dim=2).squeeze()).mean(dtype=float)
                    acc_hist.append(acc.item())
                    wandb.log({"acc": acc})

                # -- store hist --
                conn_readout_lif.append(
                {k: v.clone().detach() for k, v in fc2.named_parameters()})

                
                cur_lif_lif_hist.append(cur1_lif_lif.clone().detach())
                mem_hist['filter'].append(mem1.clone().detach())
                spk_hist['filter'].append(spk1.clone().detach())
                mem_hist['readout'].append(out.clone().detach())

        # %%
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
        acc_hist = torch.tensor(acc_hist)

        fig, ax = plt.subplots()
        ax.plot(losses)
        ax.set_title('Loss')
        ax.set_xlabel(f't/{loss_scope}')

        fname = RUN_REPORT_DIR/"loss.png"
        fig.savefig(fname)


        fig, ax = plt.subplots()
        ax.plot(conn_readout_lif['weight'][:, 0, 0])
        ax.set_title('conn_readout_lif - 0,0')
        fname = RUN_REPORT_DIR/"conn_readout_lif-0_0.png"
        fig.savefig(fname)

        fig, ax = plt.subplots()
        ax.plot(acc_hist)
        acc_mean = acc_hist.mean()
        acc_last_mean = acc_hist[-int(len(acc_hist)*.1):].mean()
        loss_last_mean = losses[-int(len(losses)*.1):].mean()
        ax.set_title(f"Accuracy| mean: {acc_mean:.2f}, last 10% mean: {acc_last_mean:.2f}")
        ax.set_xlabel("iteration/dw")
        fname = RUN_REPORT_DIR/"accuracy.png"
        fig.savefig(fname)

        wandb.log({
            "acc_mean": acc_mean,
            "acc_last_mean": acc_last_mean,
            "loss_last_mean": loss_last_mean
        })

    #%% Visualization

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
    fname = RUN_REPORT_DIR/"mempot-readout_and_reservoir.png"
    fig.savefig(fname)


    # %% LIF-LIF Connections Matrix, heatmap

    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        conn_lif_lif, cmap='bwr', cbar=True,
        linewidths=0, linecolor='grey', ax=ax)
    ax.set_xlabel('Neuron id')
    ax.set_ylabel('Neuron id')
    ax.set_title('LIF-LIF Connections Matrix')

    fname = RUN_REPORT_DIR/"reservoir-conn_matrix.png"
    fig.savefig(fname)

    # %%

    ax = src_14.visualization.generate_raster_plot(spk_hist['filter'].squeeze().T)
    ax.set_xlim(0, 2000)
    fname = RUN_REPORT_DIR/"reservoir-raster.png"
    ax.figure.savefig(fname)

    # %%

    print(f"'{RUN_REPORT_DIR}' done")



sweep_id = wandb.sweep(
    sweep=params, project=f'{EXP_DIR.name}|{RUN_PREFIX}')

wandb.agent(sweep_id, function=wandb_main)
# %%
