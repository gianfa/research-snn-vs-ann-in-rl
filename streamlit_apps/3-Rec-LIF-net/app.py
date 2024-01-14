"""
$ streamlit run 3-Rec-LIF-net/app.py


"""
import streamlit as st

st.title("snnTorch Recurrent LIF net")
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


def latek_from_tensor(tensor):
    if tensor.ndim > 2:
        raise ValueError("tensor must be 2D")
    rows = tensor.shape[0]
    cols = tensor.shape[1]

    latex = "\\begin{bmatrix}\n"
    for i in range(rows):
        row_str = " & ".join(str(tensor[i, j].item()) for j in range(cols))
        latex += row_str
        if i < rows - 1:
            latex += " \\\\\n"
    latex += "\n\\end{bmatrix}"

    return latex


params_cols = st.columns(3)

params_cols[0].write("Simulation")
dt = params_cols[0].slider(
    label="dt", min_value=0.0, max_value=0.5, value=.001, step=0.001)
n_steps = params_cols[0].slider(
    label="num of steps", min_value=0, max_value=200, value=100, step=1)

random_init_W = True
toggle_checkbox = st.checkbox("Disable Random W init for $W_{\\text{LIF\_in}}$ and $W_{\\text{LIF\_LIF}}$")
if toggle_checkbox:
    random_init_W = False
    st.write("$W_{\\text{LIF\_LIF}}$ fixed")
else:
    st.write("$W_{\\text{LIF\_LIF}}$ random init")


params_cols[1].write("Neuron parameters")
lif1_beta = params_cols[1].slider(
    label="LIF1 beta1", min_value=0.0, max_value=10.0, value=1.4, step=0.001)
lif1_threshold = params_cols[1].slider(
    label="LIF1 threshold", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
# lif2_beta = params_cols[1].slider(
#     label="LIF2 beta2", min_value=0.0, max_value=10.0, value=.27, step=0.001)
# lif2_threshold = params_cols[1].slider(
#     label="LIF2 threshold", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
lif1_lif1 = params_cols[1].slider(
    label="LIF1-LIF1 connection",
    min_value=0.0, max_value=5.0, value=1., step=.1)


reset = params_cols[1].selectbox(
    label='reset_mechanism',
    options=('subtract', 'zero', 'no reset')
)

# ---------- Input ---------------

params_cols[2].write("Input")
input_type = params_cols[2].selectbox(
    label='input type',
    options=('Cosine', 'constant', 'step', 'spiking')
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
    I_b = params_cols[2].slider(
        label="b", min_value=-10.0, max_value=10.0, value=1., step=0.2)
    I_a = params_cols[2].slider(
        label="a", min_value=-5., max_value=5., value=2., step=0.1)
    I_omega = params_cols[2].slider(
        label="omega", min_value=-3.0, max_value=3.0, value=0.3, step=0.1)
    t = torch.arange(n_steps).to(torch.float32)
    I_ = I_b + I_a * torch.cos(I_omega * t)
    st.markdown(f"$I = b + a Cos(\omega t)$ = {I_a} + Cos({I_omega} t)")

# ---------- Function ---------------

code_show = "\n".join([
    "```python\n",
    f"I = b + a * torch.cos({I_omega} * t)",
    "lif1 = snn.Leaky(",
    f"    beta={lif1_beta}, threshold={lif1_threshold},"
    f"    reset_mechanism='{reset}')",
    "# LIF1-LIF2 connection",
    f"lif1_lif2 = {lif1_lif1}",
    "```",
])

st.markdown(code_show)

# ---------- Simulation ---------------

lif1_size = 2  # layer size = N

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

def thresholded_relu(x, threshold):
    return torch.where(x > threshold, x, 0)

mem1 = torch.zeros((lif1_size)).to(torch.float) # [=] (1)
spk1 = torch.zeros((lif1_size)).to(torch.float) # [=] (1)

mem_rec = [[mem1]]  # history of membrane
spk_rec = [[spk1]]  # history of spikes

# Let's define the adjacency matrices
A_lif_in = (torch.tensor([
    [1, 0.5]
]) * torch.tensor(lif1_lif1)).to(torch.float)

A_lif_lif = torch.tensor([
    [1, 0],
    [0, 1]
]).to(torch.float)

# Connections weights
# They are not the adj matrix, since each of them is trainable as a whole.

# fixed weights, if random init is disabled
W_lif_in = torch.ones_like(A_lif_in).T
W_lif_lif = torch.ones_like(A_lif_lif)

in_lif = nn.Linear(*A_lif_in.shape)  # [=] (input, lif.size = N)
lif_lif = nn.Linear(*A_lif_lif.shape)  # [=] (lif.size, lif.size)

if not random_init_W:
    in_lif.weight = nn.Parameter(W_lif_in)
    lif_lif.weight = nn.Parameter(W_lif_lif)
else:
    W_lif_in = in_lif.weight
    W_lif_lif = lif_lif.weight

for step in range(n_steps):

    input = torch.tensor([I_[step]])  # [=] (1)

    if step % 1 == 0 or step == 0:
        # input to lif; plus connections mask
        #   in_lif(input) := input . in_lif.weight, if in_lif.weight is diagonal
        cur_in_lif = torch.mul(in_lif(input), A_lif_in)  # [=] (1, N)
    # threshold the prev activations; plus lif-lif connections mask
    # cur_lif_lif [=] (N) !! TODO: FROM HERE
    a_lif = thresholded_relu(mem1, lif1.threshold).squeeze().to(torch.float)
    cur_lif_lif = A_lif_lif @ a_lif
    
    # feed the lif
    spk1, mem1 = lif1(cur_in_lif + cur_lif_lif, mem1)

    mem_rec[0].append(mem1.squeeze())
    spk_rec[0].append(spk1.squeeze())
    print(f"DEBUG: lif1.size: {lif1_size}")
    print(f"DEBUG: mem1: {mem1.shape}")
    print(f"DEBUG: thresholded_relu:{thresholded_relu(mem1, lif1.threshold).shape}")
    print(f"DEBUG: A_lif_lif: {A_lif_lif.shape}")
    print(f"DEBUG: mem_rec: {len(mem_rec)}")
    print(f"DEBUG: spk_rec: {len(spk_rec)}")
    print(f"DEBUG: input: {input.shape}")
    print(f"DEBUG: in_lif: {in_lif}")
    print(f"DEBUG: lif_lif: {lif_lif}")
    print(f"DEBUG: thresholded_relu(mem1, lif1.threshold): {thresholded_relu(mem1, lif1.threshold).shape}")

    print(f"DEBUG: cur_in_lif: {cur_in_lif.shape}")
    print(f"DEBUG: cur_lif_lif: {cur_lif_lif.shape}")
    print(f"DEBUG: mem1: {mem1.shape}")
    print(f"DEBUG: lif_lif: {lif_lif.weight}")

# convert lists to tensors
mem_rec = [torch.stack(mi) for mi in mem_rec]
spk_rec = [torch.stack(sri) for sri in spk_rec]
print(f"DEBUG: mem_rec[0]: {mem_rec[0].shape}")
print(f"DEBUG: spk_rec[0]: {spk_rec[0].shape}")

# ---------- Visualization ---------------
# ref: `plot_cur_mem_spk` from snntorch tutorial2

# Generate Plots
colors = ['blue', 'green']
fig, ax = plt.subplots(
    3, figsize=(8, 6), sharex=True,
    gridspec_kw={"height_ratios": [1, 1, 1]}
)
ax[0].set_xlim([0, n_steps])

ax_title = None
# Plot input current
ax[0].plot(I_, c="tab:orange")
# ax[0].set_ylim([0, max(0.1, I_.max() * 1.1)])
ax[0].set_ylabel("Input Current ($I_{in}$)")
if ax_title:
    ax[0].set_title(ax_title)

# Plot membrane potential LIF1

# TODO: plot all the neurons
for i, mrec_i in enumerate(mem_rec[0].T.detach()):
    ax[1].plot(mrec_i, label=f"#{i}")
# ax[1].set_ylim([0, max(0.1, V_th * 1.2)])
ax[1].set_ylabel("Membrane Potential ($U_mem1$)")
ax[1].axhline(
    y=lif1_threshold,
    alpha=0.25, linestyle="dashed", c="black", linewidth=2
)
ax[1].legend()
ax[1].set_title('LIF1 $V_{mem}$')
ax[1].grid()

layer_id = 0
for i in range(spk_rec[layer_id].shape[1]):
    print(f"i {i}")
    spk_0_i = torch.argwhere(spk_rec[layer_id][:, i].flatten()).flatten()
    print('spk_0_i.shape:', spk_0_i.shape)
    print('spk_rec[layer_id].shape[1]', spk_rec[layer_id].shape[1])
    print(torch.ones_like(spk_0_i) * (spk_rec[layer_id].shape[1] - i))
    ax[2].scatter(
       spk_0_i,
       torch.ones_like(spk_0_i) * (spk_rec[layer_id].shape[1] - i),
       label=f"n 0", marker='|', s=int(3.e2), c=colors[i])

ax[2].set_ylim(0.5, spk_rec[layer_id].shape[1] + 0.5)
ax[2].set_yticks([1, 2])
ax[2].set_title("Spike events")
ax[2].set_xlabel("Time step")

fig.tight_layout()
st.pyplot(fig)

# ------

col1, col2 = st.columns(2)

with col2:
    st.latex("A_{LIF,in}= " + latek_from_tensor(A_lif_in.T)+"; A_{LIF,LIF}=" + latek_from_tensor(A_lif_lif))
    st.latex("W_{LIF,in}= " + latek_from_tensor(W_lif_in.T)+"; W_{LIF,LIF}=" + latek_from_tensor(W_lif_lif))
    st.latex("\sigma(x; th) = \n\\begin{cases} \nx, & \\text{if } x > \\text{th}  \\\\  \n0, & \\text{otherwise} \n\\end{cases}")
    st.latex(r"I[t]_{LIF,in} = I[t]_{in} \circ A_{LIF,in} \circ W_{LIF,in}");
    st.latex(r"a[t-1]_{LIF} = \sigma (m[t-1]_{LIF}) ")
    st.latex(r"m[t]_{LIF} = LIF(I[t]_{LIF,in} + a[t-1]_{LIF}  \cdot  W_{LIF,LIF} \circ A_{LIF,LIF})")

with col1:
    st.markdown("$A_{b, a}$: Adjacency Matrix from a to b")
    st.markdown("$W_{b, a}$: Weights Matrix from a to b")
    st.markdown("$I_{in}$: Input current  signal")
    st.markdown("$I_{LIF, in}$: Input current to the LIF layer")
    st.markdown("$m_{LIF}$: Membrane Potential of the LIF layer")
    st.markdown("$\sigma(x; th)$: Activation function")
    st.markdown("$a_{LIF}$: Activation of the LIF layer")
    st.markdown("$a \circ b$: Hadamard product between a and b. Mostly used here for masking")

# ------



exit()



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
