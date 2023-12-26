""" Signal detection with LogReg

Steps
-----
Given a signal s[t] containing freq to detect, q.
Window the signal: X = {s[t:t-w], s[t-z:t-w-z], ...}; w: window size; z: stride
Assign Y = {1 if x_i contains q, 0 otherwise }

Split X,Y -> X_test, X_train, Y_test, Y_train
Train a LorReg (X_train, Y_train)
Predict (X_test, Y_test)

Notes
-----


"""
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch

from utils import ts_gen_signal_with_sin_fragments

IMGS_DIR = Path("../imgs")
assert IMGS_DIR.exists()

# %% Generate data

(signal, labels_bounds) = ts_gen_signal_with_sin_fragments(
    baseline=.5,
    sig_coverage = 0.6)

plt.plot(range(len(signal)), signal)
for lbi in labels_bounds:
    lbi_from, lbi_to = lbi
    plt.plot(
        range(lbi_from, lbi_to),
        signal[lbi_from:lbi_to],
        color='orange')

# Create labels
labels = torch.zeros_like(signal)
for lbi in labels_bounds:
    labels[lbi[0]:lbi[1]] = 1
labels

# %% Create examples

stride = 1
wnd_size = 10
windowed_tensor = signal.unfold(0, wnd_size, stride)
windowed_labels = labels.unfold(0, wnd_size, stride)

X = windowed_tensor
y = (windowed_labels.mean(dim=1)>.5).to(int)

# %% Train/test split

train_frac = 0.80
train_limit_pos = int(len(X) * train_frac)

X_train = X[:train_limit_pos]
y_train = y[:train_limit_pos]
balance_train = y_train.mean(dtype=float)

X_test = X[train_limit_pos:]
y_test = y[train_limit_pos:]

print(f"INFO| X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"INFO| X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"INFO| balance_train: {balance_train:.2f}")

# %% Features extraction

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pca = PCA(n_components=.99)  # Mantiene il 95% della varianza
X_train_pca = pca.fit_transform(X_train_scaled)

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

print(f"INFO| X_train_scaled: {X_train_scaled.shape}")
print(f"INFO| X_test_scaled: {X_test_scaled.shape}")
print(f"INFO| X_train_pca: {X_train_pca.shape}")
print(f"INFO| X_test_pca: {X_test_pca.shape}")

# %% Train the model

clf = LogisticRegression(
    random_state=0, tol=1e-7, max_iter=10000).fit(X_train_pca, y_train)

print("INFO| clf Score on X_train pca", clf.score(X_train_pca, y_train))
print("INFO| clf Score on X_test pca", clf.score(X_test_pca, y_test))

# %% Predict

y_pred = torch.tensor(clf.predict(X_test_pca))
assert y_pred.shape == y_test.shape

test_offset = (y_train.shape[0] * stride) + wnd_size
y_pred_id_1 = torch.where(y_pred == 1)[0] + test_offset

labels_pred_x = []
labels_pred_y = []
n_features = X_train.shape[1]
for y1i_pred in y_pred_id_1:
    pos = y1i_pred
    labels_pred_x.append(torch.arange(pos, pos + n_features))
    labels_pred_y.append(X[pos: pos + n_features])

print(f"INFO|  y_pred_id_1 unique: {y_pred_id_1.unique()}")

# %% Visualize predictions

pp = len(X_train)

fig, axs = plt.subplots(2, 1)

axs[0].plot(range(len(signal)), signal)
for i, lpxi__lpyi in enumerate(zip(labels_pred_x, labels_pred_y)):
    lpxi, lpyi = lpxi__lpyi
    try:
        axs[0].plot(lpxi, lpyi, color='orange')
    except Exception as e:
        print(f"WARN| 0,0; {i}: {e}")
# plot separation between true signal and predicted one
axs[0].axvline(test_offset, linestyle='dashed', color='grey')
axs[0].legend(["Signal", "Predictions"], loc='lower right')

axs[1].plot(range(len(signal)), signal)
for i, lpxi__lpyi in enumerate(zip(labels_pred_x, labels_pred_y)):
    lpxi, lpyi = lpxi__lpyi
    try:
        axs[1].plot(lpxi, lpyi, color='orange')
    except Exception as e:
        print(f"WARN| 0,1; {i}: {e}")
axs[1].set_xlim(int(test_offset * 0.95), len(signal))
axs[1].axvline(test_offset, linestyle='dashed', color='grey')
axs[1].legend(["Signal", "Predictions"], loc='lower right')

fig.savefig(IMGS_DIR/"02-logreg-pca.png")

# %%
