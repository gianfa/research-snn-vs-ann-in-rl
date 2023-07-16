""" Visualization utilities"""
from typing import List
import matplotlib.pyplot as plt
import numpy as np

def plot_n_examples(
        X: np.ndarray, n: int, cols: int = 2, labels: List[str] = None):
    """
    

    Arguments
    ---------
    X: array-like
        signals row-wise

    Examples
    --------
    >>> X = np.random.rand(10, 100)
    >>> plot_n_examples(X, 5)
    """
    X = np.array(X)
    assert X.ndim == 2, "X must be a 2-dimensional array"
    if labels:
        assert (
            len(labels) == X.shape[0],
            "Labels must be same length as X. "
                + "They were {len(labels)} and {X.shape[0]}")
    rows = int(np.ceil(n/cols))
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(10, 8))
    print(f"rows: {rows}")
    fig.suptitle("Combination of Sin")
    up_limit = min(X.shape[0], n)
    for i, ax in enumerate(axs.ravel()):
        if i < up_limit:
            ax.plot(X[i, :], c='r')
            ax.set_title(f"$sig_{i}$")
            if labels:
                ax.set_sitle(f"{labels[i]}")
        else:
            ax.axis("off")
    fig.tight_layout()
    return axs


def get_cmap_colors(N, cmap='brg'):
    cmap = plt.cm.get_cmap(cmap)
    return [cmap(value) for value in np.linspace(0, 1, N)]