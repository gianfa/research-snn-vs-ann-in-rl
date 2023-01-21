""" Visualization utilities"""
import matplotlib.pyplot as plt
import numpy as np

def plot_n_examples(X: np.ndarray, n: int, cols: int = 2):
    """
    
    Examples
    --------
    >>> X = np.random.rand(10, 100)
    >>> plot_n_examples(X, 5)
    """
    X = np.array(X)
    assert X.ndim == 2, "X must be a 2-dimensional array"
    rows = int(np.ceil(n/cols))
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(10, 8))
    print(f"rows: {rows}")
    fig.suptitle("Combination of Sin")
    up_limit = min(X.shape[0], n)
    for i, ax in enumerate(axs.ravel()):
        if i < up_limit:
            ax.plot(X[i, :], c='r')
            ax.set_title(f"$sig_{i}$")
        else:
            ax.axis("off")
    fig.tight_layout()
    return axs