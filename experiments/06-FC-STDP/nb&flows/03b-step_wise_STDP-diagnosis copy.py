""" Get spike traces from training

Process steps
-------------
Given a FC neural network:
1. Store **layer activations** per batch at each forward step.
2. Conversion of activations into **spike traces**.
3. **STDP step** implementation
4. Weights update

Results
-------
Apparently no difference between this method and SGD + CrossEntropyLoss.
Probably due to the fact that currently spike collection stage in stdp
ignores dt=0, which is actually important in 2gen NN not time related.

Next
----
* insert performance monitors and viz
* edit spike collection algos in order to consider dt = 0. Here are some
    potential strategies for not-time-related (NTR) trainings:
    * only-0dt. Consider only dt = 0.
    * integrate over dt. If sum(dt_spk_i) with i in (1, time_window) > time_th:
        amplify, otherwise not.


"""
# %% Imports
import itertools
from functools import partial
import sys
from typing import List

from sklearn.datasets import make_classification
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path += ["..", "../..", "../../.."]
import src06.funx as src_funx
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

# %% Trial1: helpers


def get_named_layer_parameters(layer_name: str) -> dict:
    return {l_name: l_par for l_name, l_par in model.named_parameters()
            if l_name == layer_name}


def get_named_layer(layer_name: str) -> dict:
    return [l[1] for l in model.named_modules() if l[0] == layer_name][0]


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


def clear_activations(activations: dict) -> dict:
    return {name: [] for name, acts in activations.items()}


# %% Data Loading
X_orig, y_orig = make_classification(n_samples=60, n_features=10)
bs = 10  # batch size == dt_stdp in this case
train_batch, test_batch = src_funx.generate_data_batches(X_orig, y_orig, bs)

# %% Trial1: Whole training



# param: STDP
th = 0  # spike threshold
activate_STDP = True

# param: optimisation
criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01)


# # Train
model = MyNet()

# register the activations hooks
if 'hooks' in locals():
    [hook.remove() for hook in hooks]
    del hooks

activations, hooks = register_activations_hooks_to_layers(
    model, steps_to_reset=1)

for epoch in range(3):
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
            # TODO: stdp at 0 dt is ignored -> make it considered for 2gen
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
                
                print(len(activations['layer1']))
        else:
            # Backward pass e ottimizzazione
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # print(traces)

        # Training metrics
        print(f"Epoch: {epoch+1}, i: {i}, Loss: {loss.item()}")


# %%
