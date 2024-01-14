""" Multinomial Signal detection with LogReg

Steps
-----
Given a signal s[t] containing multiple freq to detect, Q: {q_1, .., q_fn}.
Window the signal: X = {s[t:t-w], s[t-z:t-w-z], ...}; w: window size; z: stride
Assign Y = {1 if x_i contains q, 0 otherwise }

Split X,Y -> X_test, X_train, Y_test, Y_train
Train a LorReg (X_train, Y_train)
Predict (X_test, Y_test)

Notes
-----

"""
# %%
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch

from utils import (ts_gen_signal_with_multisin_fragments,
    plot_signal_categs, split_tensor_by_integer_groups)

IMGS_DIR = Path("../imgs")
SAVE_FIG = False
EXP_NAME = "04-multi_logreg-feateng_200"
assert IMGS_DIR.exists()

# %% Generate data

(signal, labels_bounds) = ts_gen_signal_with_multisin_fragments(
    baseline=.5,
    sig_coverage = 0.6,
    categs=[
        {
            'freq': (3, 3),
            'length': (5, 10),
            'amplitude': 1,
            'coverage': 0.2,
        },
        {
            'freq': (17, 17),
            'length': (5, 10),
            'amplitude': 1,
            'coverage': 0.2,
        }
    ])
n_categs = len(labels_bounds)

fig, axs = plt.subplots(2, 1)
plot_signal_categs(signal, labels_bounds, ax = axs[0])
axs[0].set_title("Original Signal $X[t]$")
plot_signal_categs(signal, labels_bounds, ax = axs[1])
axs[1].set_xlim(18000, len(signal))

# Create labels: !categories starting from 1, since 0 is for baseline
labels = torch.zeros_like(signal)
for i, lbi in enumerate(labels_bounds):
    for lbi_0, lbi_1 in lbi:
        labels[lbi_0: lbi_1] = i + 1

SAVE_FIG and fig.savefig(IMGS_DIR/f"{EXP_NAME}-signal.png")

# check
#plt.plot(labels)
#labels

# %% Create examples

stride = 1
wnd_size = 200
windowed_tensor = signal.unfold(0, wnd_size, stride)
windowed_labels = labels.unfold(0, wnd_size, stride)

X = windowed_tensor
y = windowed_labels.to(int).mode(dim=1).values

# # Feature Engineering

# delay_1 = 3
# X = torch.concat((X[:-delay_1], X[delay_1:], ), dim=1)
# y = y[:-delay_1]

# delay_2 = 7
# X = torch.concat((X[:-delay_2], X[delay_2:], ), dim=1)
# y = y[:-delay_2]

# sin_transf = lambda x, period: np.sin(x / period * 2 * np.pi)

# X_sin = sin_transf(X[:, :wnd_size], 3)
# X = torch.concat((X, X_sin), dim=1)
# y = y


print("INFO| Counts: ", dict(Counter(y.tolist())))

# %% Train/test split

train_frac = 0.70
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

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

print(f"INFO| X_train_scaled: {X_train_scaled.shape}, y_train: {y_train.shape}")
print(f"INFO| X_test_scaled: {X_test_scaled.shape}, y_test: {y_test.shape}")

# %% Train the model

clf = LogisticRegression(
        multi_class='multinomial',
        random_state=0,
        tol=1e-7,
        max_iter=10000
    ).fit(X_train_scaled, y_train)

print("INFO| clf Score on X_train scaled", clf.score(X_train_scaled, y_train))
print("INFO| clf Score on X_test scaled", clf.score(X_test_scaled, y_test))

# %% Predict

y_pred = torch.tensor(clf.predict(X_test_scaled))
assert y_pred.shape == y_test.shape

test_offset = (y_train.shape[0] * stride) + wnd_size
y_pred_id = [
    torch.where(y_pred == i+1)[0] + test_offset for i in range(n_categs)]
"""Positions of labels by category
[
    [c1_id1, .., c1_idn],
    [c2_id1, .., c2_idn]
]
"""

y_pred_by_categ = split_tensor_by_integer_groups(y_pred)
print(f"INFO| len(y_pred_by_categ): {len(y_pred_by_categ)}")

# %%

colors = ['black'] + list(plt.cm.get_cmap('Accent').colors)

fig, axs = plt.subplots(2, 1)

axs[0].plot(range(len(signal)), signal, color=colors[0])  # ci = 0
for ci, y_bounds in enumerate(y_pred_by_categ):
    for yi_0, yi_1 in y_bounds:
        color_i = colors[ci]
        yi_0_, yi_1_ = yi_0 + test_offset, yi_1 + test_offset
        xi = range(yi_0_, yi_1_)
        yi = signal[yi_0_: yi_1_]
        axs[0].plot(xi, yi, color=color_i)
        
        axs[1].plot(xi, yi, color=color_i)
        axs[1].set_xlim(test_offset, len(signal))
fig.tight_layout()
SAVE_FIG and fig.savefig(IMGS_DIR/f"{EXP_NAME}-predictions.png")

# %%  Check the classification by category.

fig = plt.figure()
fig.suptitle("Predictions by category, Comparison")
for ci, y_bounds in enumerate(y_pred_by_categ):
    plt.subplot(len(y_pred_by_categ), 1, ci + 1)
    for yi_0, yi_1 in y_bounds:
        yi_0_, yi_1_ = yi_0 + test_offset, yi_1 + test_offset
        xi = range(yi_0_, yi_1_)
        yi = signal[yi_0_: yi_1_]
        plt.plot(xi, yi, color=colors[ci])
    plt.xlim(test_offset, len(signal))
    plt.title(f"Cat. {ci}")
fig.tight_layout()
SAVE_FIG and fig.savefig(IMGS_DIR/f"{EXP_NAME}-pred_by_cat-comparison.png")

# %%
