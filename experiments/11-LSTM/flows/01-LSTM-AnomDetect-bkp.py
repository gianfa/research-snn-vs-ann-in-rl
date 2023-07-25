""""

------------



"""
# %%

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
import sys

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import snntorch as snn
from sklearn.model_selection import train_test_split


sys.path += ['.', '../', '../..', '../../../']
from experimentkit_in.funx import pickle_save_dict, pickle_load
from experimentkit_in.generators.time_series import gen_simple_sin, \
    gen_random_artifacts, add_artifacts_to_signal, ts_gen_signal_shifts
import src11

# import wandb
# wandb.login()
# wandb.init(project='snnTorch-Artifacted', entity='gianfa')

ROOT = Path('../../../')
DATA_DIR = ROOT/'data'
EXP_DIR = ROOT/'experiments/11-LSTM'
EXP_DATA_DIR = EXP_DIR/'data'
EXP_REPORT_DIR = EXP_DIR/'report'

assert EXP_DIR.exists(), \
    f"CWD: {Path.cwd()}\nROOT: {ROOT.absolute()}"


# %%

# sin signal
amplitude = 1.0
frequency = 0.1
phase = 0.0

# anomalies
num_anomalies = 10
anomaly_magnitude = 5.0
# training
num_epochs = 100
batch_size = 64

DTYPE = torch.float


# %% Gen signal with random artifacts


signal_length = 10_000
num_artifacts = 200
min_artifact_length = 5
max_artifact_length = 50
min_artifact_frequency = 1
max_artifact_frequency = 10

main_signal = gen_simple_sin(
    signal_length, frequency=20, amplitude=2, base_line=2)

artifacts = gen_random_artifacts(
    num_artifacts,
    min_artifact_length,
    max_artifact_length,
    min_artifact_frequency,
    max_artifact_frequency)

dirty_signal, labels = add_artifacts_to_signal(main_signal, artifacts)

# %% visualize data set

src11.utils.plot_main_dirty_labels(main_signal, dirty_signal, labels)

# %% Generate ds collecting windowed shifts from the signal

window_size = 10

shifted_sigs = torch.Tensor(
    ts_gen_signal_shifts(dirty_signal, window_size=window_size, dim=0))
"""Shifted signals
[=](n_signals, windows_size)
"""

shifted_labels = torch.Tensor(
    ts_gen_signal_shifts(labels, window_size=window_size, dim=0))
"""Shifted labels
[=](n_labels_sigs, windows_size)
"""

# plt.plot(shifted_sigs[20, :])
fig, ax = plt.subplots()
for i in range(100):
    ax.plot(shifted_sigs[i, :] + 0.2)
ax.set_title("Example of shifted signals")

print(
    f"X_shifted_sigs.shape: {shifted_sigs.shape}" +
    f"\nshifted_labels.shape: {shifted_labels.shape}")


X = torch.tensor(shifted_sigs)  # [=] (n_sigs, window_size)
y = torch.tensor(shifted_labels)  # [=] (n_sigs, window_size)

# %%

# convert to binary labels: "is an anomaly present?"
y = y.any(dim=1).to(DTYPE)

# %% Train/Test split

test_size = 0.2
valid_size = 0.15

X_train, X_valid, X_test, y_train, y_valid, y_test = \
    src11.utils.prepare_cvset_from_X_y(
        X, y,
        test_size = test_size, valid_size = valid_size,
        DTYPE=DTYPE, v = True)
train_loader, valid_loader, test_loader = src11.utils.prepare_cvset_as_dataloaders(
        X_train, X_valid, X_test, y_train, y_valid, y_test,
        batch_size=64)

# %%

# Estimator definition
class LSTMAnomalyDetection(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMAnomalyDetection, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.binarizer = nn.Tanh()

    def forward(self, x):
        x, _= self.lstm(x)
        out = self.fc(x).squeeze()
        out = self.binarizer(out)
        return out


def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).unsqueeze(1)
        y_pred = outputs
        
        if labels.size() != outputs.size():
            raise ValueError(f"Target size {labels.size()} must be the same as input size {outputs.size()}")
        
        loss = criterion(y_pred, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_model(model, data_loader):
    model.eval()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            predicted = torch.round(outputs.squeeze())

            true_positives += torch.sum(predicted * labels).item()
            false_positives += torch.sum(predicted * (1 - labels)).item()
            false_negatives += torch.sum((1 - predicted) * labels).item()
    precision = true_positives / (true_positives + false_positives + 1e-5)
    recall = true_positives / (true_positives + false_negatives + 1e-5)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-5)
    return precision, recall, f1_score


# %% Training

# Model Definition
input_size = X_train.shape[1]
hidden_size = 30
output_size = y_train.shape[1]

model = LSTMAnomalyDetection(input_size, hidden_size, output_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

train_losses = []
valid_losses = []
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer)
    valid_loss = evaluate_model(model, valid_loader)[2]
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f} | Valid F1-Score: {valid_loss:.4f}")


precision, recall, f1_score = evaluate_model(model, test_loader)
print(f"\nTest Precision: {precision:.4f} | Test Recall: {recall:.4f} | Test F1-Score: {f1_score:.4f}")

# %%

src11.utils.plot_performance(train_losses, valid_losses)

# %% 

y_pred, X_test_2, y_test_2 = src11.utils.evaluate_model_2(model, valid_loader)

src11.utils.plot_anomalies_over_signal(X_test_2, y_test_2, title='Test predictions')

# %% not used

def plot_examples(X, y, predictions = None, title = "X and y", ax = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(X, label="Artifacted signal")
    ax.plot(y, "r", label="Labels")
    if predictions is not None:
        anomaly_indices = np.where(predictions == 1)[0]
        ax.scatter(
            anomaly_indices, X[anomaly_indices],
            color="g", label="Predicted Anomalies")
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("A")
    ax.legend()


src11.utils.plot_from_shifted(X_train[:400], y_train[:400], title='Training')


# Plot del segnale originale
src11.utils.plot_signal(X)


# %%
# Plot del confronto tra labels e predizioni
# all_predictions = torch.round(model(X_test)).detach().numpy()
# plot_examples(X_test, y_test, all_predictions)
