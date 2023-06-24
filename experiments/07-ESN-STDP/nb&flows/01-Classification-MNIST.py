""" MNIST classification with ESN + STDP

Q: Is it possible to achieve an improvement in the classification
    performance of an ESN by applying STDP?

Premises and observations:
    0. Classification on images is something almost out-of-scope for ESN
    1. Each image is shown as single example made up of P features, where
        P is the total number of pixels.
    2. STDP cannot occur during reservoir recursion, since it would alter
        the activations to which the readout is mapped.

Exp Observations:
    1. It is prone to exploding weights since STDP will amplify indefinitely.
        -> Normalization before update.
    2. 

        
Ref:
1. SCHAETTI, Nils; SALOMON, Michel; COUTURIER, Raphaël. Echo state networks-based reservoir computing for mnist handwritten digits recognition. In: 2016 IEEE Intl Conference on Computational Science and Engineering (CSE) and IEEE Intl Conference on Embedded and Ubiquitous Computing (EUC) and 15th Intl Symposium on Distributed Computing and Applications for Business Engineering (DCABES). IEEE, 2016. p. 484-491.
"""
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
import torch

sys.path += ["..", "../..", "../../.."]
from stdp.estimators import BaseESN
from stdp import funx as stdp_f

ROOT = Path('../../../')
DATA_DIR = ROOT/'data'
EXP_DIR = ROOT/'experiments/07-ESN-classification'
EXP_DATA_DIR = EXP_DIR/'data'
EXP_REPORT_DIR = EXP_DIR/'report'

assert EXP_DIR.exists(), \
    f"CWD: {Path.cwd()}\nROOT: {ROOT.absolute()}"
# %%

# # Data Loading
X, y = load_digits(return_X_y=True)  # noqa

encoder = OneHotEncoder(sparse_output=False)
y_oh = encoder.fit_transform(y.reshape(-1, 1)).squeeze()
print(y_oh.shape)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y_oh, test_size=0.2, random_state=42, shuffle=True)

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


input_size = X_train.shape[1]
reservoir_size = 100
output_size = 10
esn = BaseESN(
    input_size,
    reservoir_size,
    output_size,
    connections = (stdp_f.generate_simple_circle_connections_mask(
        (reservoir_size, reservoir_size)) > 0).int()
)

# Training
states = esn.train(X_train, y_train)

# %%
y_test_int = y_test.argmax(axis=1)

y_out = esn.predict(X_test)
y_pred_int = np.argmax(
    torch.softmax(y_out, dim=1).numpy(), axis=1)

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score


acc = accuracy_score(y_test_int, y_pred_int)
prec = precision_score(y_test_int, y_pred_int, average='macro')
f1 = f1_score(y_test_int, y_pred_int, average='macro')

# Scores
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"F1: {f1:.2f}")

# %%

t0 = time.time()
th = 0
W_hist = [esn.W.clone()]
for epoch in range(1):
    raster = (states > th).to(int)[:, -20:]
    # update hidden connections
    reservoir_weights = esn.W.clone()
    reservoir_connections = esn.connections

    # print(layer.weight.mean())
    new_W = stdp_f.stdp_step(
        reservoir_weights,
        connections=reservoir_connections,
        raster=raster,
        spike_collection_rule = stdp_f.all_to_all,
        dw_rule = "sum",
        max_delta_t=4,
    )
    # TODO: Issue: connections must be ternary and sparse

    bad_diff = (
        (new_W !=0 ).to(int) - (reservoir_connections != 0).to(int)
        ).sum()
    assert bad_diff == 0
    print("STDP executed")
W_hist.append(new_W)

print(f"t: {time.time() - t0:.0f}")


# %% ---------- many steps ---------

t0 = time.time()
th = 0
W_hist = [esn.W.clone()]
scores = dict(
        accuracy = [ round(accuracy_score(y_test_int, y_pred_int), 3)],
        precision = [ round(
            precision_score(y_test_int, y_pred_int, average='macro'), 2)],
        f1 = [ round(f1_score(y_test_int, y_pred_int, average='macro'), 2)],
    )
for epoch in range(10):
    raster = (states > th).to(int)[:, -50:]
    # update hidden connections
    reservoir_weights = esn.W.clone()
    reservoir_connections = esn.connections

    new_W = stdp_f.stdp_step(
        reservoir_weights,
        connections=reservoir_connections,
        raster=raster,
        spike_collection_rule = stdp_f.all_to_all,
        dw_rule = "sum",
        max_delta_t=4,
    )
    # Normalize weights
    # new_W /= new_W.max()
    new_W = ((new_W / new_W.max()) * 2 - 1) * esn.connections

    if epoch % 2 == 0:
        new_W = 0.5 * new_W + \
            torch.randn_like(esn.W) * 0.5 * esn.connections

    # TODO: Issue: connections must be ternary and sparse

    bad_diff = (
        (new_W !=0 ).to(int) - (reservoir_connections != 0).to(int)
        ).sum()
    assert bad_diff == 0
    W_hist.append(new_W)
    
    print("STDP executed")
    print(f"Step t: {time.time() - t0:.0f}")

    # # Re-train #
    esn.W = new_W
    states = esn.train(X_train, y_train)

    # # Evaluate #
    y_test_int = y_test.argmax(axis=1)
    y_out = esn.predict(X_test)
    y_pred_int = np.argmax(
        torch.softmax(y_out, dim=1).numpy(), axis=1)

    scores['accuracy'].append(
        round(accuracy_score(y_test_int, y_pred_int), 3))
    scores['precision'].append(
        round(precision_score(y_test_int, y_pred_int, average='macro'), 2))
    scores['f1'].append(
        round(f1_score(y_test_int, y_pred_int, average='macro'), 2))

    print(f"Accuracy: {scores['accuracy'][-1]:.2f}")
    print(f"Precision: {scores['precision'][-1]:.2f}")
    print(f"F1: {scores['f1'][-1]:.2f}\n")


# %%

stdp_f.plot_most_changing_node_weights_and_connection(
    torch.stack(W_hist), n_top_weights=5)

# %% Plot ESN connections

fig, ax = plt.subplots()
ax.set(
    title='Reservoir Connections'
)

stdp_f.connections_to_digraph(esn.connections, ax=ax)

# %% Plot W_hist
import seaborn as sns

fig, axs = plt.subplots(
    len(W_hist), 2, figsize=(8, 18))

for i, weights in enumerate(W_hist):
    #axs[i, 0].hist(weights.view(-1).numpy(), bins=40)
    sns.boxplot(
        data=weights[weights > 0].ravel().numpy(), orient="h", ax=axs[i, 0])
    axs[i, 0].set(
        title=f'Nonzero W, step #{i+1}',
        xlabel='Value',
        ylabel='Count'
    )
    sns.heatmap(
        weights.numpy(),
        ax=axs[i, 1],
        cmap='bwr', annot=False, cbar=True)
    axs[i, 1].set(
        title=f'W, step #{i+1}',
    )
fig.subplots_adjust()
fig.suptitle('Weights distribution along STDP steps')
fig.tight_layout()
plt.show()

# %%

fig, ax = plt.subplots()
for name, score in scores.items():
    ax.plot(score, '-o', label=name)
ax.set(
    title='scores'
)
ax.grid()
ax.legend()

# %%