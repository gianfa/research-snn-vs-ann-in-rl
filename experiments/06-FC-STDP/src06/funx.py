from functools import partial
from typing import List

from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ML

def generate_data_batches(X, y, batch_size, shuffle=True):
    """
    
    Example
    -------
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    bs = 32
    train_loader, test_loader = generate_data_batches(X, y, bs)
    for batch_features, batch_labels in train_loader:
        print(batch_features.shape, batch_labels.shape)
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# STDP Helpers

def get_named_layer_parameters(model: nn.Module, layer_name: str) -> dict:
    return {l_name: l_par for l_name, l_par in model.named_parameters()
            if l_name == layer_name}


def get_named_layer(model: nn.Module, layer_name: str) -> dict:
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
