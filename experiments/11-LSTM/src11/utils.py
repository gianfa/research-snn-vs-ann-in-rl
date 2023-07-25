from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch


def prepare_cvset_from_X_y(
        X, y, test_size = 0.2, valid_size = 0.15, v = True, DTYPE=float):

    X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=test_size, shuffle=False)

    valid_size = 0.15 / (1 - test_size)
    X_train, X_valid, y_train, y_valid = \
        train_test_split(X_train, y_train, test_size=valid_size, shuffle=False)
    
    assert (
        len(X) - 10 < len(X_train) + len(X_valid) + len(X_test) < len(X) + 10)

    # X_train [=] (n_fragments, fragment_len)
    X_train = X_train.to(DTYPE)
    X_valid = X_valid.to(DTYPE)
    X_test = X_test.to(DTYPE)
    y_train = y_train.to(DTYPE).unsqueeze(1)
    y_valid = y_valid.to(DTYPE).unsqueeze(1)
    y_test = y_test.to(DTYPE).unsqueeze(1)

    v and print(
        X_train.shape, y_train.shape,
        X_valid.shape, y_valid.shape,
        X_test.shape, y_test.shape)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def prepare_cvset_as_dataloaders(
        X_train, X_valid, X_test, y_train, y_valid, y_test, batch_size=32):
    # # convert all to DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True)
    return train_loader, valid_loader, test_loader


def plot_signal(signal, ax = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(signal)
    ax.set_title("Original Signal")
    ax.set_xlabel("t")
    ax.set_ylabel("A")


def plot_from_shifted(X, y, title='', ax = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(X[:, 0], label="X")
    ax.plot(y[:, 0], label='y')
    ax.set_title(title)
    ax.legend()
    ax.grid()
    return ax


def plot_shifted(
    shifted_sigs,   # (n_signals, window_size)
    n = 8,  # number of signals to show
    offset = 3,  # steps between a signal and the next
    skip = 5  # signals to skip before to start
):
    fig, axs = plt.subplots(n, sharex=True)
    for i in range(n):
        sig_i = skip + i * offset
        axs[i].plot(shifted_sigs[sig_i, :])
        axs[i].set_yticklabels([])
        axs[i].set_ylabel(sig_i, rotation=0)
    fig.suptitle('Shifted signal')
    fig.tight_layout()
    return axs


from matplotlib.colors import LinearSegmentedColormap

def binary_cmap():
    colors = [(1, 1, 1), (0, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('binary_cmap', colors, N=2)
    return cmap


def plot_binary_vectors(
        y: torch.tensor, figsize=None, title: str = '') -> Tuple[plt.Axes]:
    """Plot a list of binary vectors

    Parameters
    ----------
    y : torch.tensor
        (n_vectors, n_features)
    title : str, optional
        The plot title, by default ''

    Returns
    -------
    Tuple[plt.Axes]
        The collection of axes, each containing a vector.
    """
    figsize_ = figsize if figsize else (7, 1.5 * y.shape[0])
    fig, axes = plt.subplots(
        y.shape[0], 1, figsize=figsize_, sharex=True)
    fig.subplots_adjust(left=0.2)
    fig.suptitle(title)

    for i in range(y.shape[0]):
        y_i = y[i].reshape(1, -1)

        axes[i].imshow(
            #y_i, cmap='gray', aspect='auto')
            y_i, cmap=binary_cmap(),
            aspect='auto',
            extent=[0, 10, 0, 1],
            origin='lower',
            vmin=0, vmax=1)
        for j in np.where(y_i.flatten())[0]:
            axes[i].axvline(j + 0.5, color='gray')

        axes[i].set_title(title)
        axes[i].set_ylabel(f'$y_{{{i+1}}}$ ', rotation=0)
        axes[i].set_yticks([])

    axes[i].set_xlabel('Index')
    fig.tight_layout()
    return axes



def plot_main_dirty_labels(main_signal, dirty_signal, labels):
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(main_signal)
    axs[0].set_title('main_signal')
    axs[1].plot(dirty_signal)
    axs[1].set_title('dirty_signal')
    # axs[2].plot(labels)
    labels_idx = np.argwhere(labels)
    axs[2].plot(
        labels_idx, np.ones_like(labels_idx),
        marker='|', markersize=22, linestyle="")
    axs[2].set_ylim(0.9)
    axs[2].set_title('anomaly labels indices')
    axs[2].get_yaxis().set_visible(False)
    fig.tight_layout()
    return axs


def plot_performance(train_losses, valid_losses):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Model performance")
    
    axs[0].plot(train_losses, label="Train Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid()
    
    axs[1].plot(valid_losses, label="Valid F1-Score")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("F1-Score")
    axs[1].legend()
    axs[1].grid()
    axs[1].set_ylim(top=1)
    
    return axs


def plot_anomalies_over_signal(X, y, title='', axs = None):
    if axs is None:
        _, axs = plt.subplots(
            2, 1, figsize=(10, 4), sharex=True,
            gridspec_kw={'height_ratios': [2, 1]})
    fig = axs[0].figure

    axs[0].plot(X[:,0], color='black')
    lab_1_idxs = np.where(y)[0].flatten()
    split_idxs = np.array([0] + np.where(np.diff(lab_1_idxs)>1)[0].tolist() )
    split_idxs_pairs = [(i, j) for i, j in zip(split_idxs[:-1], split_idxs[1:])]

    for i, j in  split_idxs_pairs:
        xx = lab_1_idxs[i+2: j]
        yy = X[xx, 0]
        axs[0].plot(xx, yy, color='orange')
    
    axs[0].legend(['signal', 'anomalies'])
    axs[0].set_title(title)
    axs[0].grid()

    axs[1].plot(
        np.where(y), np.ones_like(np.where(y)), marker="|", color='grey')
    axs[1].set_yticklabels([])
    axs[1].grid()

    return axs


def evaluate_model_2(model, dloader):
    y_pred = []
    y_test_2 = []
    X_test_2 = []
    model.eval()
    with torch.no_grad():
        for xi_test, yi_test in dloader:
            outputs = model(xi_test)
            predicted = torch.round(outputs.squeeze())
            y_pred.append(predicted)
            y_test_2.append(yi_test)
            X_test_2.append(xi_test)

    y_pred = torch.stack(y_pred)
    y_test_2 = torch.stack(y_test_2).view(-1)
    X_test_2 = torch.stack(X_test_2)
    X_test_2 = X_test_2.view(-1, X_test_2.shape[-1])
    return y_pred, X_test_2, y_test_2