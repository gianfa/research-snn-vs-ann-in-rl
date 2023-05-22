""" ESN: STDP in theReservoir plus SGD in the Readout

ESN
Connection matrix

- STDP on middle layer + SGD on output gradient (liike Yussuf).
    It's actually automatically done, since in the ESN definition the only
    layer with gradient is the readout.

ISSUE: not sure wheter `raster` is consistent with the stored activations

Process steps
-------------
Given a FC neural network:
1. Store **layer activations** per batch at each forward step.
2. Conversion of activations into **spike traces**.
3. **STDP step** implementation
4. Weights update

Results
-------


"""
# %% Imports
import itertools
from functools import partial
from pathlib import Path
import sys
from typing import List

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path += ["..", "../..", "../../.."]
import src06.funx as src_funx
from stdp import funx as stdp_f
from stdp.estimators import ESN
from stdp.spike_collectors import all_to_all, nearest_pre_post_pair

from experimentkit_in.metricsreport import MetricsReport

ROOT = Path('../../../')
DATA_DIR = ROOT/'data'
EXP_DIR = ROOT/'experiments/06-FC-STDP'
EXP_DATA_DIR = EXP_DIR/'data'
EXP_REPORT_DIR = EXP_DIR/'report'

# %% Helpers

def register_activations_hooks_to_layers(
        model: nn.Module,
        steps_to_reset: int or None = None,
        layers: List[str] = None,
        v: bool = False) -> \
    List[torch.utils.hooks.RemovableHandle]:

    layer_names = [
            name for name, _ in model.named_modules() if len(name) > 0] \
            if layers is None else layers

    def init_activations() -> dict:
        return {name: [] for name in layer_names}

    def get_activations(layer, input, output, label: str):
        if steps_to_reset and len(activations[label]) >= steps_to_reset:
            # activations[label].clear()
            activations[label] = activations[label][-1:]
        activations[label].append(output)

    activations = init_activations()

    hooks = []
    for name, layer in model.named_modules():
        if name in layer_names:
            hooks.append(
                layer.register_forward_hook(
                partial(get_activations, label=name)))
            if v: print(f"Registered hook to layer: {name}")
    return activations, hooks



# %% Data Loading

examples_dim = 20
target_dim = 1

# # Data Generation
X, y, sk_coeffs = make_regression(
    n_samples=100,
    n_features=examples_dim,
    n_informative=5,
    n_targets=1,
    noise=5,
    coef=True,
    random_state=1
)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train).unsqueeze(1)
y_test = torch.tensor(y_test).unsqueeze(1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# %% Model definition

# # Define ESN
input_size = examples_dim
hidden_size = 100
output_size = target_dim
spectral_radius = 0.9

# -
model = ESN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    spectral_radius=spectral_radius)


layer_names = [name for name, _ in model.named_modules() if len(name) > 0]

# %% Trial1: Whole training


# param: STDP
th = 0  # spike threshold
activate_STDP = True

# param: optimisation
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=4e-2)


# # Train

# register the activations hooks
if 'hooks' in locals():
    [hook.remove() for hook in hooks]
    del hooks

activations, hooks = register_activations_hooks_to_layers(
    model, layers=['W'], steps_to_reset=5)

train_mr = MetricsReport()
valid_mr = MetricsReport()
loss_hist = []
test_loss_hist = []
traces = []
t_start = time.time()
for epoch in range(13):
    for i, X_i_Y_i in enumerate(zip(X_train, y_train)):
        X_i, Y_i = X_i_Y_i
        X_i, Y_i = X_i.to(torch.float32), Y_i.to(torch.float32)
        if X_i.ndim == 1: X_i = X_i.unsqueeze(0)  # expand as a row
        if Y_i.ndim == 1: Y_i = Y_i.unsqueeze(0)  # expand as a row
        # output [=] (output.shape, out_features) = (bs, out_features)
        out = model(X_i, activation=torch.tanh)
        loss = criterion(out, Y_i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if activate_STDP:
            if epoch > 0 and epoch % 4 == 0: # check ste_to_reset in register
                traces = src_funx.activations_to_traces(activations, th)
                raster = traces.T # torch.stack(traces).squeeze().T
                spks = stdp_f.raster_collect_spikes(
                    raster, collection_rule=all_to_all)
                # update hidden connections
                layer_name = 'W'
                layer = src_funx.get_named_layer(model, layer_name)
                # print(layer.weight.mean())
                new_layer_W = stdp_f.stdp_step(
                    getattr(model, layer_name).weight,
                    connections=None,
                    raster=raster,
                    spike_collection_rule = all_to_all,
                    dw_rule = "sum",
                    max_delta_t=4,
                )
                # update weights normalizing them
                new_layer_W = nn.Parameter(new_layer_W)#/new_layer_W.max())
                layer.weight = new_layer_W
                assert getattr(model, layer_name).weight.equal(new_layer_W)
                print("STDP executed")
        
        
        # print(traces)

        # Training metrics
        loss_hist.append(loss.detach())
        print(f"Epoch: {epoch+1}, i: {i}, Loss: {loss.item()}")

# %%
fig, ax = plt.subplots()
ax.plot(loss_hist)
ax.set_title('Loss')
ax.set_xlabel('# iteration')
ax.set_yscale('log')
ax.grid()
fig.savefig(EXP_DATA_DIR/"ESN-STDP-loss.png")
fig.savefig(EXP_REPORT_DIR/"ESN-STDP-loss.png")

fig, ax = plt.subplots()
y_pred = model.predict(X_test)
ax.plot(y_test, label='y_test')
ax.plot(y_pred, label='y_pred')
ax.set(
    title='Test signal vs Predicted signal',
    xlabel='t'
)
ax.grid()
ax.legend()
fig.savefig(EXP_DATA_DIR/"ESN-STDP-compare-output_pred.png")
fig.savefig(EXP_REPORT_DIR/"ESN-STDP-compare-output_pred.png")

# %%
