""" STDP batch-wise

* performance metrics was added.
* T_dw lookup now is 1 at dt = 0 and 0 else where, since data is
    not time-related
* weights are normalized during STDP


Process steps
-------------
Given a FC neural network:
1. Store **layer activations** per batch at each forward step.
2. Conversion of activations into **spike traces**.
3. **STDP step** implementation
4. Weights update

Results
-------

The network does not seem to learn much.
What it could influence:
- STDP scans inter-layer activations over time, and here the data
    are not correlated over time
- even if we dwell on the batch step as an atomic time sequence,
    there is no way to amplify the weights involved in the prediction
    of a single example.
It would be much simpler to turn it into a simple reinforcement:
    if y_pred=y_true => amplify weights
- Check dimensions consistency between layer units and the raster given to STDP

"""
# %% Imports
import itertools
from functools import partial
import sys
from typing import List

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path += ["..", "../..", "../../.."]
import src06.funx as src_funx
from stdp import funx as stdp_f
from stdp.spike_collectors import all_to_all, nearest_pre_post_pair

from experimentkit_in.metricsreport import MetricsReport


# %% Helpers

def register_activations_hooks_to_layers(
        model: nn.Module,
        steps_to_reset: int or None = None, v: bool = False) -> \
    List[torch.utils.hooks.RemovableHandle]:

    def init_activations() -> dict:
        return {
            name: [] for name, _ in model.named_modules() if len(name) > 0}

    def get_activations(layer, input, output, label: str):
        if steps_to_reset and len(activations[label]) >= steps_to_reset:
            activations[label].clear()
        activations[label].append(output)

    activations = init_activations()

    hooks = []
    for name, layer in model.named_modules():
        if len(name) > 0:
            hooks.append(
                layer.register_forward_hook(
                partial(get_activations, label=name)))
            if v: print(f"Registered hook to layer: {name}")
    return activations, hooks



# %% Data Loading
X_orig, y_orig = make_classification(n_samples=60, n_features=4)
bs = 10  # batch size == dt_stdp in this case
train_batch, test_batch = src_funx.generate_data_batches(X_orig, y_orig, bs)


# %% Model definition

class MyNet(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super(MyNet, self).__init__()
        self.layer1 = nn.Linear(n_in, 5)
        self.layer2 = nn.Linear(5, 3)
        self.layer3 = nn.Linear(3, n_out)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x


model = MyNet(n_in=4, n_out=1)


layer_names = [name for name, _ in model.named_modules() if len(name) > 0]

# %% Trial1: Whole training


# param: STDP
th = 0.5  # spike threshold
activate_STDP = True

# param: optimisation
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# # Train

# register the activations hooks
if 'hooks' in locals():
    [hook.remove() for hook in hooks]
    del hooks

activations, hooks = register_activations_hooks_to_layers(
    model, steps_to_reset=1)

train_mr = MetricsReport()
valid_mr = MetricsReport()
loss_hist = []
test_loss_hist = []
t_start = time.time()
for epoch in range(100):
    for i, Xi_yi in enumerate(train_batch):
        Xi, yi = Xi_yi
        out = model(Xi)
        yi_pred = (out > 0.5).to(int)

        loss = criterion(out, yi.unsqueeze(1).float())
        
        if not activate_STDP:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            # STDP
            traces = src_funx.activations_to_traces(activations, th)
            raster = traces.clone()
            spks = stdp_f.raster_collect_spikes(
                raster, collection_rule=all_to_all)
            # TODO: stdp at 0 dt is ignored -> make it considered for 2gen
            for layer_name in layer_names:
                layer = src_funx.get_named_layer(model, layer_name)
                # print(layer.weight.mean())
                new_layer_W = stdp_f.stdp_step(
                    getattr(model, layer_name).weight,
                    connections=None,
                    raster=raster,
                    spike_collection_rule = all_to_all,
                    dw_rule = "sum",
                    max_delta_t=4,
                    time_related=False
                )
                # update weights normalizing them
                new_layer_W = nn.Parameter(new_layer_W)#/new_layer_W.max())
                layer.weight = new_layer_W
                assert getattr(model, layer_name).weight.equal(new_layer_W)
                
                #print(len(activations['layer1']))
        
        
        # print(traces)

        # Training metrics
        train_mr.append(yi.unsqueeze(1), yi_pred)
        loss_hist.append(loss.detach())
        print(f"Epoch: {epoch+1}, i: {i}, Loss: {loss.item()}")


fig, ax = plt.subplots()
ax.plot(loss_hist)
ax.set_title('Loss')

train_mr.plot_metrics(None, smooth=True, show_highlights=True)
# %%
