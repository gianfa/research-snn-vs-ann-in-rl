""" Get spike traces from training



Next
----

stdp acts between steps

if (spk_post[t] + spk_pre[t-1]) == 2 : amplify
if (spk_pre[t] + spk_post[t-1]) == 2 : attenuate




"""
# %% Imports
import itertools
from functools import partial
import sys

from typing import List
import torch
import torch.nn as nn
import torch.optim as optim

sys.path += ["..", "../..", "../../.."]
from stdp import funx as stdp_f
from stdp.spike_collectors import all_to_all

# %% Register get-activations hook

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 3)
        self.layer3 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


model = MyNet()


layer_names = [name for name, _ in model.named_modules() if len(name) > 0]
# activations = {name: None for name in layer_names}

input_tensor = torch.randn(1, 10)

def register_activations_hooks_to_layers(model: nn.Module, v: bool = False) -> \
    List[torch.utils.hooks.RemovableHandle]:
    activations = {
        name: [] for name, modules in model.named_modules() if len(name) > 0}

    def get_activations(layer, input, output, label: str):
        activations[label].append(output)

    hooks = []
    for name, layer in model.named_modules():
        if len(name) > 0:
            hooks.append(
                layer.register_forward_hook(
                partial(get_activations, label=name)))
            if v: print(f"Registered hook to layer: {name}")
    return activations, hooks

activations, hooks = register_activations_hooks_to_layers(model)


out = model(input_tensor)

# Hooks remove
for hook in hooks:
    hook.remove()

# See the activations
for layer_name, activation in activations.items():
    print(f"{layer_name} acts: {activation}")

# %% Get spikes from one step
# Threshold to get the spikes
#   Here it assumes that all the units have
#       the same activation -> same threshold -> spike
th = 0
step_id = 0
spks = {li: act[step_id] > th for li, act in activations.items()}
print(spks)

# %% Trial2: helper

def get_named_layer_parameters(layer_name: str) -> dict:
    return {l_name: l_par for l_name, l_par in model.named_parameters()
            if l_name == layer_name}

def get_named_layer(layer_name: str) -> dict:
    return [l[1] for l in model.named_modules() if l[0] == layer_name][0]


def activations_as_tensors(activations: dict):
    return {layer_name: torch.vstack(acts)
                for layer_name, acts in activations.items()}


def traces_ensure_are_3_dim(traces: dict) -> dict:
    return {
        name:(trace if trace.ndim==3 else trace.unsqueeze(2))
        for name, trace in traces.items()}


def traces_contract_to_2_dim(traces: dict) -> dict:
    """{LAYER: tensor(nbatches x nsteps, nunits)}"""
    return {
        name: trace.view(-1, trace.size(2))
        for name, trace in traces_ensure_are_3_dim(traces).items()}


def layertraces_to_traces(traces: dict) -> dict:
    return torch.vstack(
        [trace.T for trace in traces_contract_to_2_dim(traces).values()])


def activations_to_traces(activations: dict, th: float) -> torch.Tensor:
    acts_tens = activations_as_layertraces(activations, th)
    return layertraces_to_traces(acts_tens)


def activations_as_layertraces(activations: dict, th: float) -> dict:
    return {layer_name: torch.vstack(acts) > th
                for layer_name, acts in activations.items()}


def activations_to_traces(activations: dict, th: float) -> torch.Tensor:
    return  torch.hstack( list(
                    activations_as_layertraces(activations, th).values()))




# %% Trial2: 1-Step Store training spikes



bs = 10
dt_stdp = 5
th = 0.5
train_batch = torch.randn(bs, 10)

model = MyNet()
activations, hooks = register_activations_hooks_to_layers(model)

for i, data_targets in enumerate(train_batch):
    out = model(input_tensor)
    print(out)


activations = {layer_name: torch.stack(acts).squeeze()
    for layer_name, acts in activations.items()}
# remove hoooks
[hook.remove() for hook in hooks]

# Get spike traces
traces = {li: acts > th for li, acts in activations.items()}
""" {LAYER: tensor(nunits x steps)} """

traces

# %% Trial3: Whole training

import src06.funx as src_funx
from sklearn.datasets import make_classification
import torch.nn.functional as F

# param: STDP
th = 0  # spike threshold
activate_STDP = False

# param: optimisation
criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Data Loading
X_orig, y_orig = make_classification(n_samples=60, n_features=10)
bs = 10  # batch size == dt_stdp in this case
train_batch, test_batch = src_funx.generate_data_batches(X_orig, y_orig, bs)

# # Train
model = MyNet()

# register the activations hooks
activations, hooks = register_activations_hooks_to_layers(model)

for epoch in range(2):
    for i, X_y in enumerate(train_batch):
        X, y = X_y
        out = model(X)
        y_pred = F.softmax(out, dim=0)

        loss = criterion(y_pred.squeeze().float(), y)
        
        # STDP
        traces = activations_to_traces(activations, th)
        raster = traces.clone()
        spks = stdp_f.raster_collect_spikes(raster, collection_rule=all_to_all)

        if activate_STDP:
            for layer_name in layer_names:
                layer = get_named_layer(layer_name)
                # print(layer.weight.mean())
                new_layer1_W = stdp_f.stdp_step(
                    getattr(model, layer_name).weight,
                    connections=None,
                    raster=raster,
                    spike_collection_rule = all_to_all,
                    dw_rule = "prod",
                    max_delta_t=4,
                )
                layer.weight =  nn.Parameter(new_layer1_W)
                assert getattr(model, layer_name).weight.equal(
                    nn.Parameter(new_layer1_W))
        else:
            # Backward pass e ottimizzazione
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # print(traces)

        # Stampa delle metriche di allenamento
        print(f"Epoch: {epoch+1}, i: {i}, Loss: {loss.item()}")




# %%
