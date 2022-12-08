# %%
"""STDP Implementation trials, on LIFs

Decription
----------
Contains tests of STDP done on LIF networks.


References
----------
1. https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html#define-the-network
2. https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html
3. `NEFTCI, Emre O.; MOSTAFA, Hesham; ZENKE, Friedemann. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks. IEEE Signal Processing Magazine, 2019, 36.6: 51-63. <https://core.ac.uk/download/pdf/323310555.pdf>`_


Theory
------
Learning: Loss
- Surrogate gradient approach.
- Spike-Operator approach. -> ∂S/∂U ϵ {0,1}


Output decoding
- we take the output layer as a logit layer, where each neuron
    fires for a specific class.
- we choose "rate coding" as a decoding strategy. This means that
    we'll expect the neuron assigned to the right class to fire
    at higher freq during the specific class showing up.
    - ONE way is to increase U_th of the neuron and decrease the others ones
        - softmax p_i[t] = exp(U_i[t] / sum( exp(U_j[t]), 0, C))
        - L_{CE}[t] = sum( y_i log(p_i[t]), 0,C )
        - => L_{CE} = sum( L_{CE}[t] ,t)

"""

# %% Imports

import time
import os
import snntorch as snn  # noqa
from snntorch import spikeplot as splt  # noqa
from snntorch import spikegen  # noqa

import torch  # noqa
import torch.nn as nn  # noqa
from torch.utils.data import DataLoader  # noqa
from torchvision import datasets, transforms  # noqa

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import itertools  # noqa


# %% Helper functions


def batch_accuracy(data, targets):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())
    return acc


def print_batch_accuracy(data, targets, train=False):
    acc = batch_accuracy(data, targets)

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")


def train_printer(
    data, targets,
    epoch, counter, iter_counter, loss_hist,
        test_loss_hist, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")



# %% Data Loading

# dataloader arguments
batch_size = 128
data_path = '../data'


if not os.path.isdir(data_path):
    raise Exception("Data directory not found")

dtype = torch.float
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(
    data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(
    data_path, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(
    mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)


# %% # Network Definition

# Network Architecture
num_inputs = 28 * 28
num_hidden = 100
num_outputs = 10

# Temporal Dynamics
num_steps = 25
beta = 0.95


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


# Load the network onto CUDA if available
net = Net().to(device)


loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

"""
    After only one iteration, the loss should have decreased and accuracy
should have increased. Note how membrane potential is used to calculate the
cross entropy loss, and spike count is used for the measure of accuracy. It
is also possible to use the spike count in the loss.
"""


# %% # Training Loop

num_epochs = 1

loss_hist = []
test_loss_hist = []

counter = 0
t_start = time.time()
# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data.view(batch_size, -1))

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)
        
        if iter_counter % 50 or iter_counter == 0:
            print("| epoch | iteration | Train loss | b. accuracy |")
            print("|-------|-----------|------------|")
        
        acc = batch_accuracy(data, targets)
        print(f"| {epoch} | {iter_counter} | {loss_val.item():.3f} | {acc} |")

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = net(test_data.view(batch_size, -1))

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 50 == 0:
                train_printer(
                    data, targets, epoch, counter, iter_counter,
                    loss_hist, test_loss_hist, test_data, test_targets)
            counter += 1
            iter_counter += 1

print(f"Elapsed time: {time.time() - t_start}s")


# Visualize
# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

#  Evaluation on Test
total = 0
correct = 0

# drop_last switched to False to keep all samples
test_loader = DataLoader(
    mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

with torch.no_grad():
    net.eval()
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        test_spk, _ = net(data.view(data.size(0), -1))

        # calculate total accuracy
        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")
# %%
