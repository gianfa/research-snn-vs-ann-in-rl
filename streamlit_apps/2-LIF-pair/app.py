"""
$ streamlit run 2-LIF-pair/app.py
"""
import streamlit as st

st.title("snnTorch 2LIFs tutorial")
"""
This is a web app to play with
[snnToch](https://snntorch.readthedocs.io/en/latest/index.html)
"""
import snntorch as snn  # noqa
from snntorch import spikeplot as splt  # noqa
from snntorch import spikegen  # noqa

import torch  # noqa
import torch.nn as nn  # noqa

import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa

from tut_utils import *  # noqa


params_cols = st.columns(3)

params_cols[0].write("Simulation")
dt = params_cols[0].slider(
    label="dt", min_value=0.0, max_value=0.5, value=.001, step=0.001)
n_steps = params_cols[0].slider(
    label="num of steps", min_value=0, max_value=200, value=100, step=1)


params_cols[1].write("Neuron parameters")
lif1_beta = params_cols[1].slider(
    label="LIF1 beta1", min_value=0.0, max_value=10.0, value=1.4, step=0.001)
lif1_threshold = params_cols[1].slider(
    label="LIF1 threshold", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
lif2_beta = params_cols[1].slider(
    label="LIF2 beta2", min_value=0.0, max_value=10.0, value=.27, step=0.001)
lif2_threshold = params_cols[1].slider(
    label="LIF2 threshold", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
lif1_lif2 = params_cols[1].slider(
    label="LIF1-LIF2 connection",
    min_value=0.0, max_value=5.0, value=1., step=.1)


reset = params_cols[1].selectbox(
    label='reset_mechanism',
    options=('subtract', 'zero', 'no reset')
)

# ---------- Input ---------------

params_cols[2].write("Input")
input_type = params_cols[2].selectbox(
    label='input type',
    options=('constant', 'step', 'spiking', 'Cosine')
)
if input_type == 'constant':
    I_bl = params_cols[2].slider(
        label="base line I", min_value=0.0, max_value=3.0, value=.3, step=0.1)
    I_max = params_cols[2].slider(
        label="max I", min_value=0.0, max_value=3.0, value=.3, step=0.1)
    t_start = params_cols[2].slider(
        label="t_start", min_value=0, max_value=300, value=10, step=1)

    if t_start > n_steps:
        raise Exception("t_start must be lower than n_steps")
    I_ = torch.ones(n_steps, dtype=torch.float32) * I_bl
    I_[t_start:] = I_max

if input_type == 'step':

    I_bl = params_cols[2].slider(
        label="base line I", min_value=0.0, max_value=3.0, value=.3, step=0.1)
    I_max = params_cols[2].slider(
        label="max I", min_value=0.0, max_value=3.0, value=.3, step=0.1)
    step_duration = params_cols[2].slider(
        label="step duration", min_value=0, max_value=20, value=10, step=1)

    # Add additional step start times interactively
    steps = []
    if 'n_rows' not in st.session_state:
        st.session_state.n_rows = 1

    add = params_cols[2].button(label="add")

    if add:
        st.session_state.n_rows += 1
        st.experimental_rerun()

    for i in range(st.session_state.n_rows):
        step_start = params_cols[2].slider(
                label=f"step n.{i} start time",
                min_value=0, max_value=n_steps, value=0, step=1)
        if step_start > n_steps:
            raise Exception("t_start must be lower than n_steps")
        steps.append(step_start)

    I_ = torch.ones(n_steps, dtype=torch.float32) * I_bl
    for step_i in steps:
        I_[step_i: step_i + step_duration] = I_max

if input_type == 'spiking':
    I_bl = params_cols[2].slider(
        label="base line I", min_value=0.0, max_value=3.0, value=.0, step=0.1)
    I_max = params_cols[2].slider(
        label="max I", min_value=0.0, max_value=3.0, value=.3, step=0.1)
    frate = params_cols[2].slider(
        label="Mean input firing rate",
        min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    I_ = spikegen.rate_conv(torch.ones((n_steps)) * I_max * frate + I_bl)

if input_type == 'Cosine':
    # 1+cos(omega t)
    # omega
    I_a = params_cols[2].slider(
        label="a", min_value=-10.0, max_value=10.0, value=1., step=0.2)
    I_omega = params_cols[2].slider(
        label="omega", min_value=-3.0, max_value=3.0, value=0.05, step=0.1)
    t = torch.arange(n_steps).to(torch.float32)
    I_ = I_a + torch.cos(I_omega * t)
    st.markdown(f"$I = a + Cos(\omega t)$ = {I_a} + Cos({I_omega} t)")

# ---------- Function ---------------

code_show = "\n".join([
    "```python\n",
    "lif1 = snn.Leaky(",
    f"    beta={lif1_beta}, threshold={lif1_threshold},"
    f"    reset_mechanism='{reset}')",
    "lif2 = snn.Leaky(",
    f"    beta={lif2_beta}, threshold={lif2_threshold},"
    f"    reset_mechanism='{reset}')",
    "# LIF1-LIF2 connection",
    f"lif1_lif2 = {lif1_lif2}",
    "```",
])

st.markdown(code_show)

# ---------- Simulation ---------------

lif1 = snn.Leaky(
    beta=lif1_beta,  # membrane potential decay rate, clipped. May be a tensor.
    threshold=lif1_threshold,
    spike_grad=None,  # surrogate gradient function
    init_hidden=False,
    inhibition=False,  # If `True`, suppresses all spiking other than the neuron with the highest state
    learn_beta=False,  # enable learnable beta
    learn_threshold=False,  # enable learnable threshold
    reset_mechanism="zero",  # {'subtract', zero', 'none'}
    state_quant=False,  # If `True` as well as `init_hidden=True`, states are returned
    output=False
)

lif2 = snn.Leaky(
    beta=lif2_beta,  # membrane potential decay rate, clipped. May be a tensor.
    threshold=lif2_threshold,
    spike_grad=None,  # surrogate gradient function
    init_hidden=False,
    inhibition=False,  # If `True`, suppresses all spiking other than the neuron with the highest state
    learn_beta=False,  # enable learnable beta
    learn_threshold=False,  # enable learnable threshold
    reset_mechanism="zero",  # {'subtract', zero', 'none'}
    state_quant=False,  # If `True` as well as `init_hidden=True`, states are returned
    output=False
)

def thresholded_relu(x, threshold):
    return torch.where(x > threshold, x, 0)

mem1 = torch.zeros(1)
mem2 = torch.zeros(1)
mem_rec = [[], []]  # history of membrane
spk_rec = [[], []]  # history of spikes

lif1_lif2 = torch.tensor([lif1_lif2])

for step in range(n_steps):
    cur_in1 = I_[step]
    spk1, mem1 = lif1(cur_in1, mem1)

    cur_in2_1 = lif1_lif2 * thresholded_relu(mem1, lif1.threshold)
    
    spk2, mem2 = lif2(cur_in2_1, mem2)

    mem_rec[0].append(mem1)
    mem_rec[1].append(mem2)
    spk_rec[0].append(spk1)
    spk_rec[1].append(spk2)

# convert lists to tensors
mem_rec = [torch.stack(mi) for mi in mem_rec]
spk_rec = [torch.stack(sri) for sri in spk_rec]


# ---------- Visualization ---------------
# ref: `plot_cur_mem_spk` from snntorch tutorial2

# Generate Plots
colors = ['blue', 'green']
fig, ax = plt.subplots(
    4, figsize=(8, 6), sharex=True,
    gridspec_kw={"height_ratios": [1, 1, 1, 1]}
)
ax[0].set_xlim([0, n_steps])

ax_title = None
# Plot input current
ax[0].plot(I_, c="tab:orange")
ax[0].set_ylim([0, max(0.1, I_.max() * 1.1)])
ax[0].set_ylabel("Input Current ($I_{in}$)")
if ax_title:
    ax[0].set_title(ax_title)

# Plot membrane potential LIF1
ax[1].plot(mem_rec[0], c=colors[0])
# ax[1].set_ylim([0, max(0.1, V_th * 1.2)])
ax[1].set_ylabel("Membrane Potential ($U_mem1$)")
ax[1].axhline(
    y=lif1_threshold,
    alpha=0.25, linestyle="dashed", c="black", linewidth=2
)
ax[1].set_title('LIF1 $V_{mem}$')
ax[1].grid()

# Plot membrane potential LIF2
ax[2].plot(mem_rec[1], c=colors[1])
# ax[2].set_ylim([0, max(0.1, V_th * 1.2)])
ax[2].set_ylabel("Membrane Potential ($U_mem2$)")
ax[2].axhline(
    y=lif2_threshold,
    alpha=0.25, linestyle="dashed", c="black", linewidth=2
)
ax[2].set_title('LIF2 $V_{mem}$')
ax[2].grid()


plt.xlabel("Time step")


for i in range(len(spk_rec)):
    spk_0_i = torch.argwhere(spk_rec[i].flatten())
    ax[3].scatter(
       spk_0_i,
       torch.ones_like(spk_0_i) * (len(spk_rec)-i),
       label=f"n 0", marker='|', s=int(3.e2), c=colors[i])
# ax[3].set_xlim(range_to_plot.start, range_to_plot.stop)
ax[3].set_ylim(0.5, len(spk_rec)+0.5)
ax[3].set_title("Spike events")
ax[3].set_yticks([1, 2])
ax[3].set_ylim([0, len(spk_rec) + 1])
# Plot output spike using spikeplot
# splt.raster(spk_rec[0], ax[3], s=400, c="black", marker="|")
# vline = None
# if vline:
#     ax[2].axvline(
#         x=vline,
#         ymin=0,
#         ymax=6.75,
#         alpha=0.15,
#         linestyle="dashed",
#         c="black",
#         linewidth=2,
#         zorder=0,
#         clip_on=False,
#     )
fig.tight_layout()
st.pyplot(fig)
