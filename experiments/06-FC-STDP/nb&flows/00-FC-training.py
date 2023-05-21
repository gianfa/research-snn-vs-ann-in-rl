# %%
""" Bare FC training

"""
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
from stdp.spike_collectors import all_to_all

from experimentkit_in.metricsreport import MetricsReport


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 3)
        self.layer3 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x

model = MyNet()

# param: optimisation
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# # Train



X_orig, y_orig = make_classification(n_samples=60, n_features=10)
bs = 10  # batch size == dt_stdp in this case
train_batch, test_batch = src_funx.generate_data_batches(X_orig, y_orig, bs)


loss_hist = []
test_loss_hist = []
t_start = time.time()
for epoch in range(3):
    for i, Xi_yi in enumerate(train_batch):
        Xi, yi = Xi_yi
        out = model(Xi)
        yi_pred = out

        loss = criterion(yi_pred, yi.unsqueeze(1).float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Training metrics
        loss_hist.append(loss.detach())
        print(f"Epoch: {epoch+1}, i: {i}, Loss: {loss.item()}")


plt.plot(loss_hist)
# %%