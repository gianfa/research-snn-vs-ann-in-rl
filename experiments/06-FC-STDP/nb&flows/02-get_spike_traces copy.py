""" Get spike traces from training




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


# %% Trial2: Store training spikes

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

model = MyNet()

bs = 10  # batch size
dt_stdp = 5  #
th = 0.5  # spike threshold
X_orig, y_orig = make_classification(n_samples=50, n_features=10)
train_batch, test_batch = src_funx.generate_data_batches(X_orig, y_orig, bs)


criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01)
activations, hooks = register_activations_hooks_to_layers(model)

for epoch in range(1):
    for i, X_y in enumerate(train_batch):
        X, y = X_y
        out = model(X)
        y_pred = F.softmax(out, dim=0)

        loss = criterion(y_pred.squeeze().float(), y)
        
        # Backward pass e ottimizzazione
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Stampa delle metriche di allenamento
        print(f"Epoch: {epoch+1}, i: {i}, Loss: {loss.item()}")



activations = {layer_name: torch.stack(acts).squeeze()
    for layer_name, acts in activations.items()}

# remove hoooks
[hook.remove() for hook in hooks]

# Get spike traces
traces = {li: acts > th for li, acts in activations.items()}
""" {LAYER: tensor(nbatches, nsteps, nunits)} """


def traces_ensure_are_3_dim(traces: dict) -> dict:
    return {
        name:(trace if trace.ndim==3 else trace.unsqueeze(2))
        for name, trace in traces.items()}


def traces_contract_to_2_dim(traces: dict) -> dict:
    """{LAYER: tensor(nbatches x nsteps, nunits)}"""
    return {
        name: trace.view(-1, trace.size(2))
        for name, trace in traces_ensure_are_3_dim(traces).items()}


def traces_to_tensor(traces: dict) -> dict:
    return torch.vstack(
        [trace.T for trace in traces_contract_to_2_dim(traces).values()])


def activations_to_tensor(activations: dict) -> torch.Tensor:
    return traces_to_tensor({li: acts > th for li, acts in activations.items()})



activations_to_tensor(activations)