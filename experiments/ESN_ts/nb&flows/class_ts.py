# %%
import sys
sys.path += ["..", "../..", "../../.."]
from typing import Callable, List, Tuple  # noqa

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ( # noqa
    accuracy_score, precision_score, f1_score, confusion_matrix)
import torch  # noqa
from torch import nn
from torch.optim.optimizer import Optimizer

from stdp.estimators import ESN
from experimentkit_in.visualization import plot_n_examples

def train(
    model: nn.Module,
    X_y: List[Tuple[torch.TensorType, int]],
    optimizer: Optimizer,
    criterion: Callable = nn.BCEWithLogitsLoss(),
    epochs: int = 10) -> tuple:

    hist = {'loss': [], 'out': [], 'epochs_idx': [], 'y_pred': [], 'y': []}
    iteration = 0
    for epoch in range(epochs):
        hist['epochs_idx'].append(iteration)
        hist['y_pred'].append([])
        hist['y'].append([])
        for X, y in X_y:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out.float(), y.float())
            optimizer.step()

            y_pred = int((out > 0).int())
            hist['loss'].append(loss)
            hist['out'].append(out)
            hist['y_pred'][-1].append(y_pred)
            hist['y'][-1].append(y)
            iteration += 1
    
    hist['y_pred'] = torch.vstack([torch.tensor(x) for x in hist['y_pred']])
    hist['y'] = torch.vstack([torch.stack(x).T for x in hist['y']])
    hist['out'] = torch.vstack(hist['out'])
    hist['loss'] = torch.stack(hist['loss'])
    return model, hist

# Data creation
n_examples_per_class = 60 
X_y = (
    [(
        amp_i * torch.sin(torch.arange(100))[:, None],
        torch.tensor([0], dtype=torch.int8))
            for amp_i in range(n_examples_per_class)] +
        [(
            amp_i * (torch.cos(torch.arange(100)) * torch.sin(torch.arange(100)))[:, None],
            torch.tensor([1], dtype=torch.int8))
                for amp_i in range(n_examples_per_class)]
)
X_y = [X_y[i] for i in torch.randperm(len(X_y)).tolist()]
train_len = int(len(X_y) * 0.8)
print(f"train length: {train_len}")
X_y_train = X_y[:train_len]
X_y_test = X_y[train_len:]

examples = np.vstack([X_y[i][0].numpy().T for i in range(6)])
plot_n_examples(examples, 6)

# %% # Model creation

input_size = 1
hidden_size = 10
output_size = 1
spectral_radius = 0.9

# criterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()
esn = ESN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    spectral_radius=spectral_radius,
    )

# Consider this
#  https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch
optimizer = torch.optim.Adam(esn.W_out.parameters(), weight_decay=1e-3)

# %% Training

epochs = 2

esn, hist = train(
    model=esn, X_y=X_y_train, optimizer=optimizer, epochs=epochs,
    criterion = criterion)

# %% Viz

fig, ax = plt.subplots()
ax.plot(hist['loss'].detach().numpy())
ax.set_title('ESN Training Loss')
ax.set_xlabel('iterations')


import pandas as pd
aa=pd.DataFrame(torch.tensor(hist['loss'].detach().numpy()))
aa.describe()
# %%

def batch_metrics(metrics, y_pred, y):
    return torch.tensor([
        metrics(ypi, yi) for ypi, yi in zip(y_pred, y)])


acc_x_batch = batch_metrics(accuracy_score, hist['y_pred'], hist['y'])

f1_x_batch = batch_metrics(f1_score, hist['y_pred'], hist['y'])

fig, axs = plt.subplots(1, 2)
axs[0].plot(acc_x_batch)
axs[0].set_title(f'accuracy: {round(acc_x_batch.mean().item(), 2)}')
axs[1].plot(f1_x_batch)
axs[1].set_title(f'f1: {round(f1_x_batch.mean().item(), 2)}')
fig.tight_layout()

# %%

test_losses = []
with torch.no_grad():
    test_loss = 0.0
    esn.eval()
    for X, y in X_y_test:
        out = esn(X)
        loss = criterion(out, y.float())
        test_losses.append(loss)
        #test_loss += loss.item() * X.shape[0]
  
# mean loss
#test_loss /= len(X_y_test)
test_losses = torch.tensor(test_losses)
print('Test Loss: {:.6f}\n'.format(torch.mean(test_losses)))

plt.plot(test_losses)
plt.title("Loss")
plt.xlabel("iterations")
# %%

plt.plot(hist['out'].detach().numpy())
plt.title("Out")
plt.xlabel("iterations")

# %%
