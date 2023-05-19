# %%

from functools import partial
from typing import List
import torch
import torch.nn as nn

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


net = MyNet()

input_tensor = torch.randn(1, 10)


activations = {  # will collect the activations
    'L1': None,
    'L2': None,
    'L3': None,
}

# Hook the function to the activations step
def get_activations(layer, input, output, label: str):
    activations[label] = output

hooks=[]
hooks.append(net.layer1.register_forward_hook(partial(get_activations, label='L1')))
hooks.append(net.layer2.register_forward_hook(partial(get_activations, label='L2')))
hooks.append(net.layer3.register_forward_hook(partial(get_activations, label='L3')))



# Esecuzione dell'inoltro attraverso la rete
out = net(input_tensor)

# Hooks remove
for hook in hooks:
    hook.remove()

# See the activations
for layer_name, activation in activations.items():
    print(f"{layer_name} acts: {activation}")

# Threshold to get the spikes
spks = {li: act > 0 for li, act in activations.items()}
print(spks)


for name, param in net.named_parameters():
    if 'weight' in name:
        W = torch.empty_like(param.data)
        print(W.shape)

# %%


def get_activations_from_layers(net: nn.Module, v: bool = False) -> \
    List[torch.utils.hooks.RemovableHandle]:
    """ Get activations from each layer of a neural network.

    Parameters
    ----------
    net : (nn.Module)
        The neural network model.

    Returns
    -------
    hooks : List[torch.utils.hooks.RemovableHandle]
        List of hook handles.

    Examples
    --------
    >>> class MyNet(nn.Module):
    ...     def __init__(self):
    ...         super(MyNet, self).__init__()
    ...         self.layer1 = nn.Linear(10, 5)
    ...         self.layer2 = nn.Linear(5, 3)
    ...         self.layer3 = nn.Linear(3, 1)

    ...     def forward(self, x):
    ...         x = torch.relu(self.layer1(x))
    ...         x = torch.relu(self.layer2(x))
    ...         x = self.layer3(x)
    ...         return x
    >>> net = MyNet()
    >>> hooks = get_activations_from_layers(net)
    >>> hooks
    [<torch.utils.hooks.RemovableHandle at 0x11527e220>,
    <torch.utils.hooks.RemovableHandle at 0x1152668e0>,
    <torch.utils.hooks.RemovableHandle at 0x1151c4160>]
    """
    activations = {
        name: None for name, modules in net.named_modules() if len(name) > 0}

    def get_activations(layer, input, output, label: str):
        activations[label] = output

    hooks = []
    for name, layer in net.named_modules():
        if len(name) > 0:
            hooks.append(
                layer.register_forward_hook(
                partial(get_activations, label=name)))
            if v: print(f"Registered hook to layer: {name}")
    return activations, hooks

activations, hooks = get_activations_from_layers(net)
hooks