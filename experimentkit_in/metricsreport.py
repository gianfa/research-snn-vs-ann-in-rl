import copy
from typing import List, Tuple

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, confusion_matrix
)

from .funx import *

class MetricsReport():
    """
    Examples
    --------
    >>> mr = MetricsReport()
    >>> mr.append([1,0,1], [1,1,0])
    >>> mr.append([1,0,1], [1,1,0])
    >>> print(len(mr))
    2
    >>> m, n = mr.get_metrics(return_names=True)

    # You can easily visualize metrics using pandas
    >>> import pandas as pd
    >>> ms, n = mr.get_metrics(return_names=True)
    >>> pd.DataFrame(ms, columns=n)
    """

    def __init__(
        self,
        feature_names: List[str] = None,
        target_names: List[str] = None
    ):
        self.X = []
        self.y_pred = []
        self.y_true = []
        self.metrics = {
            'accuracy': [],
            'precision': [],
            'f1': [],
            'n_samples': []
        }
        self.feature_names = feature_names
        self.target_names = target_names

    def __len__(self):
        return len(self.y_pred)

    def __str__(self):
        txt = f"MetricsReport\n{self.__len__()} rows\n"
        return txt


    # TODO: review and doc
    # self.get_true_pred()
    def __getitem__(self, idx):
        """
        Returns
        -------
        _ : List[Torch.Tenstor]
            [y_true[idx], y_pred[idx]]


        Example
        -------
        >>> mr[:3][0].shape
        torch.Size([3, 128])
        >>> mr[:3][1].shape
        torch.Size([3, 128])
        """
        return [
            torch.stack(self.y_true[idx], dim=0),
            torch.stack(self.y_pred[idx], dim=0)]

    def copy(self):
        return copy.deepcopy(self)

    def append(self, y_true, y_pred):
        if np.shape(y_true) != np.shape(y_pred):
            raise ValueError(
                "y_true and y_pred must have the same shape. " +
                f"They were: {y_true.shape} and {y_pred.shape}")
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)
        avg = 'macro'
        if type(y_true) == torch.Tensor:
            y_true = y_true.detach().numpy()
        if type(y_pred) == torch.Tensor:
            y_pred = y_pred.detach().numpy()
        self.metrics['n_samples'].append(len(y_true))
        self.metrics['accuracy'].append(accuracy_score(y_true, y_pred))
        self.metrics['precision'].append(precision_score(y_true, y_pred, average=avg))
        self.metrics['f1'].append(f1_score(y_true, y_pred, average=avg))

    def get_metrics(self, return_names: bool=False, return_frame: bool=False):
        metrics = np.stack(list(self.metrics.values()), axis=0).T
        if return_frame:
            return pd.DataFrame(metrics, columns=list(self.metrics.keys()))
        if return_names:
            return metrics, list(self.metrics.keys())
        return metrics
    
    def get_metrics_stats(
            self, return_names: bool=False, return_frame: bool=False) -> dict:
        metrics, names = self.get_metrics(return_names=True)
        stats = {
            'mean': metrics.mean(axis=0),
            'std': metrics.std(axis=0),
            'min': metrics.min(axis=0),
            'max': metrics.max(axis=0)
        }
        if return_frame:
            return pd.DataFrame(stats, columns=names)
        if return_names:
            return (stats, names)
        return stats

    def get_true_pred(self):
        # check shapes
        if len(set([len(yi) for yi in self.y_true])) == 1:
            yt = np.stack(self.y_true, axis=0).flatten()
        else:
            yt = np.hstack(self.y_true)

        if len(set([len(yi) for yi in self.y_pred])) == 1:
            yp = np.stack(self.y_pred, axis=0).flatten()
        else:
            yp = np.hstack(self.y_pred)

        return yt, yp
    
    def compute_confusion_matrix(
            self, over_all: bool = False, normalize: str =None) -> np.ndarray:
        if over_all:
            yt, yp = self.get_true_pred()
        else:
            yt, yp = self.y_true[-1], self.y_pred[-1]
        labels = None
        if self.target_names and len(self.target_names)>0:
            labels = self.target_names
        return confusion_matrix(yt, yp, labels=labels, normalize=normalize);
    
    def plot_confusion_matrix(
        self,
        title: str ="Confusion matrix",
        normalize: str =None,
        **kwargs):
        cm = self.compute_confusion_matrix(normalize=normalize)
        xticklabels = "auto"
        yticklabels = "auto"
        if self.feature_names and len(self.feature_names)>0:
            xticklabels = self.feature_names
        if self.target_names and len(self.target_names)>0:
            yticklabels = self.target_names
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(title)
        sns.heatmap(
            data=cm,
            annot=True,
            xticklabels=xticklabels, yticklabels=yticklabels, ax=ax, **kwargs)
        return ax
    
    def plot_metrics(
            self,
            ax: plt.Axes =None,
            ylim: Tuple[int, int] =(0,1),
            smooth: bool =False,
            show_highlights =True
        ) -> plt.Axes:
        if not ax:
            fig, ax = plt.subplots()
        fig = ax.figure

        for name, m in self.metrics.items():
            if name != 'n_samples':
                if smooth == True:
                    ax.plot(moving_average(m)[1:-1], label=name)
                else:
                    ax.plot(m, label=name)
                if show_highlights:
                    ax.axhline(
                        max(m), label=f"max {name}",
                        linewidth=0.5, linestyle='dashed', color='k')
        
        

        ax.set(
            title='Metrics',
            ylim=ylim
        )
        ax.legend()
        ax.grid(True)
        return ax