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


TODOs
-----
* plot at least 1 Res weight comparison between no-STDP and STDP.
* implement L.Sq Optimisation
* Lorenz oscillator
    * forecast t+1, t+1, t+3, ..
    * https://scipython.com/blog/the-lorenz-attractor/
    * x or z or w as a target
    * https://matplotlib.org/stable/gallery/mplot3d/lorenz_attractor.html
"""
# %% Imports
import itertools
from functools import partial
from pathlib import Path
import sys
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
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
    n_samples=5000,
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
"""
* STDP update in ESN weight must happen after each epoch, since it should not
    interfere with recurrent execution, because it will alter the projection
    to the readout.
"""

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
W_hist = [model.W.weight.data]
loss_hist = []
test_loss_hist = []
traces = []
t_start = time.time()
connections = stdp_f.generate_sparse_matrix(
    model.W.weight.shape, 0.4, torch.ones).to_dense()

for epoch in range(3):
    X_train, y_train = X_train.to(torch.float32), y_train.to(torch.float32)
    if X_train.ndim == 1: X_train = X_train.unsqueeze(0)  # expand as a row
    if y_train.ndim == 1: y_train = y_train.unsqueeze(0)  # expand as a row
    # output [=] (output.shape, out_features) = (bs, out_features)
    out = model(X_train, activation=torch.tanh)
    loss = criterion(out, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    
        
        
    # print(traces)

    # Training metrics
    loss_hist.append(loss.detach())
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    if activate_STDP:
        if epoch > 0 and epoch % 1 == 0: # check step_to_reset in register
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
                connections=connections,
                raster=raster,
                spike_collection_rule = all_to_all,
                dw_rule = "sum",
                max_delta_t=4,
            )
            # TODO: Issue: connections must be ternary and sparse

            # update weights normalizing them
            new_layer_W = nn.Parameter(new_layer_W)#/new_layer_W.max())
            layer.weight = new_layer_W
            assert getattr(model, layer_name).weight.equal(new_layer_W)
            print("STDP executed")
    W_hist.append(model.W.weight.data)

W_hist = torch.stack(W_hist)

# %% Plot most changing weights

stdp_f.plot_most_changing_node_weights_and_connection(W_hist, n_top_weights=5)

# %%

n_top_weights = 5
# W_hist = torch.stack(W_hist)
abs_diff = np.abs( np.diff(W_hist.detach().numpy(), axis=0) ).sum(axis=0)
if abs_diff.sum() == 0:
    print("INFO: No difference between W along hist")
else:
    flat_indices = np.argsort(abs_diff, axis=None)[-n_top_weights:]
    row_indices, col_indices = np.unravel_index(flat_indices, abs_diff.shape)

fig, ax = plt.subplots()
for ri, ci in zip(row_indices, col_indices):
    ax.plot(W_hist[:, ri, ci], "-o", label=f"{ri} -> {ci}")
ax.grid()
ax.legend()
ax.set(
    title=f'Top {n_top_weights} changing weights',
    xlabel=f"# epoch"
)

# Plot Topology
nodes = set(col_indices.tolist() + row_indices.tolist()); print(nodes)
edges = [(row, col) for row, col in zip(row_indices, col_indices)]
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
pos = nx.spring_layout(G)  # Posizionamento dei nodi nel layout
fig, ax = plt.subplots()
nx.draw_networkx(G, pos, with_labels=True, arrows=True, ax=ax)
ax.set(title='Topology of the most changing weights')

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Stampa i risultati
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")


# %%
