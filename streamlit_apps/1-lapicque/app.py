"""
$ streamlit run 1-lapicque/app.py
"""
import streamlit as st

st.title("snnTorch tutorial")
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
R = params_cols[1].slider(
    label="R", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
C = params_cols[1].slider(
    label="C", min_value=0.0, max_value=0.6, value=5e-3, step=0.001)
V_th = params_cols[1].slider(
    label="V_th", min_value=0.0, max_value=3.0, value=.5, step=0.1)

reset = params_cols[1].selectbox(
    label='reset_mechanism',
    options=('subtract', 'zero', 'no reset')
)

# ---------- Input ---------------

params_cols[2].write("Input")
input_type = params_cols[2].selectbox(
    label='input type',
    options=('constant', 'step', 'spiking')
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

# ---------- Function ---------------

lapicque_func = "\n".join([
    "```python\n",
    "snn.Lapicque(",
    f"    R={R}, C={C}, time_step={dt}, threshold={V_th},",
    f"    reset_mechanism='{reset}')",
    "```",
    f"tau={R*C}",
])

st.markdown(lapicque_func)

# ---------- Simulation ---------------

lif1 = snn.Lapicque(
        R=R, C=C, time_step=dt, threshold=V_th,
        reset_mechanism=reset)

mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec = [mem]
spk_rec = [spk_out]
for step in range(n_steps):
    spk_out, mem = lif1(I_[step], mem)
    mem_rec.append(mem)
    spk_rec.append(spk_out)
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)


# ---------- Visualization ---------------
# ref: `plot_cur_mem_spk` from snntorch tutorial2

# Generate Plots
fig, ax = plt.subplots(
    3, figsize=(8, 6), sharex=True,
    gridspec_kw={"height_ratios": [1, 1, 0.4]}
)
ax[0].set_xlim([0, n_steps])

ax_title = None
# Plot input current
ax[0].plot(I_, c="tab:orange")
ax[0].set_ylim([0, max(0.1, I_.max() * 1.1)])
ax[0].set_ylabel("Input Current ($I_{in}$)")
if ax_title:
    ax[0].set_title(ax_title)

# Plot membrane potential
ax[1].plot(mem_rec)
ax[1].set_ylim([0, max(0.1, V_th * 1.2)])
ax[1].set_ylabel("Membrane Potential ($U_mem$)")
ax[1].axhline(
    y=V_th, alpha=0.25, linestyle="dashed", c="black", linewidth=2
)
plt.xlabel("Time step")

# Plot output spike using spikeplot
splt.raster(spk_rec, ax[2], s=400, c="black", marker="|")
vline = 109
if vline:
    ax[2].axvline(
        x=vline,
        ymin=0,
        ymax=6.75,
        alpha=0.15,
        linestyle="dashed",
        c="black",
        linewidth=2,
        zorder=0,
        clip_on=False,
    )
plt.ylabel("Output spikes")
plt.yticks([])
st.pyplot(fig)
