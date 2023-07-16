import argparse
from functools import partial
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import yaml

from experimentkit_in.funx import pickle_save_dict, pickle_load
from experimentkit_in.generators.time_series import gen_lorenz
from experimentkit_in.visualization import get_cmap_colors

def load_yaml(): ...

def argparse_config():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", dest="config", required=False)
    args = arg_parser.parse_args()

    params = load_yaml(fpath=args.config)
    return params

# Analysis #

def evaluate_regression_report(y_true, y_pred):
    return dict(
        mse = mean_squared_error(y_true, y_pred),
        mae = mean_absolute_error(y_true, y_pred),
        r2 = r2_score(y_true, y_pred)
    )

def plot_evaluate_regression_report(
        y_true, y_pred, title_prefix=''):
    rep = evaluate_regression_report(y_true, y_pred)
    rep_txt = "<br>".join(
        [f"{str(k).capitalize()}: {v:.2f}" for k, v in rep.items()])
    title = f"{title_prefix}<br>{rep_txt}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_pred.ravel(), name='y<sub>pred</sub>'))
    fig.add_trace(go.Scatter(y=y_true.ravel(), name='y<sub>test</sub>'))
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5
        },
        xaxis_title='t',
        yaxis_title='Y',
        legend=dict(x=0, y=1)
    )

    # fig.show()
    return fig


## EXP ##

def expt_generate_new_lorenz_data(
    example_len = 10000,
    test_size = 0.2,
    valid_size = 0.15,
    recompute = False,
    ds_path = None,
    shift = 12,  # forecasted delay
    s = 12, r = 30, b = 2.700,
    time_last = True, # put time dimension as last
    v: bool = False
):
    if shift < 1:
        raise Exception(f"shift must be greater than 1!")

    ds_path = Path(ds_path)
    if ds_path.exists() and not recompute:
        ds = pickle_load(ds_path)
    else:
        ds = gen_lorenz(n_steps=example_len, s=s, r=r, b=b)
        pickle_save_dict(ds_path, ds)

    X = ds[:-shift]
    y = ds[shift:, 0]

    v and print(X.shape, y.shape)
    assert X.shape[0] == y.shape[0], f"{X.shape}[0] != {y.shape}[0]"

    # train/test split

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, shuffle=False)

    valid_size = valid_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = \
        train_test_split(X_train, y_train, test_size=valid_size, shuffle=False)

    X_train = torch.tensor(X_train, dtype=torch.float)
    X_valid = torch.tensor(X_valid, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float).unsqueeze(1)
    y_valid = torch.tensor(y_valid, dtype=torch.float).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float).unsqueeze(1)

    if time_last:
        X_train = X_train.T
        X_valid = X_valid.T
        X_test = X_test.T
        y_train = y_train.T
        y_valid = y_valid.T
        y_test = y_test.T

    print(
        X_train.shape, y_train.shape,
        X_valid.shape, y_valid.shape,
        X_test.shape, y_test.shape)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


## Plot ##

def plot_eigenvalues_tensor(
        evals: torch.Tensor,
        plot_circle: bool = True,
        x_label: str = 'Re',
        y_label: str = 'Im',
        ax: plt.Axes = None,
    ) -> plt.Axes:
    """Plot eigenvalues tensor

    Parameters
    ----------
    evals : torch.Tensor
        2D tensor.

    Returns
    -------
    plt.Axes
    """

    if evals.ndim != 2:
        raise Exception()
    color_cycle = get_cmap_colors(evals.shape[0])
    
    if not ax:
        _, ax = plt.subplots()
    for i in range(evals.shape[0]):
        evals_i = evals[i]
        ax.plot(
            evals_i.real, evals_i.imag,
            'o', color=color_cycle[i],
            markersize=0.5, label=f"{i}")
    ax.axhline(0, color='gray')
    ax.axvline(0, color='gray')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('Eigenvalues')

    # Plot a reference circle
    if plot_circle:
        n_dots = 1000
        angles = np.linspace(0, 2*np.pi, n_dots, endpoint=False)
        circle_x = 1 * np.cos(angles)
        circle_y = 1 * np.sin(angles)
        ax.plot(
            circle_x, circle_y, '.',
            markersize=0.2, color='grey', label=f"r=1")

    ax.legend()

    return ax


def plot_compare_df_via_boxplot(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    names: List[str] = ['df1', 'df2'],
    xlabel: str ='',
    ylabel: str ='',
    title: str ='Comparison',
    ylim: tuple = None,
    ax: plt.Axes = None,
) -> plt.Axes:

    if not ax:
        _, ax = plt.subplots()



    # add a 'Origin' column to the dataframes
    df1.loc[:, 'DF'] = names[0]
    df2.loc[:, 'DF'] = names[1]

    # make the two one
    df_combined = pd.concat([df1, df2])

    # convert to 'long' format
    df_combined = df_combined.melt(
        id_vars=['DF'], var_name='Column', value_name='Value')

    sns.boxplot(
        data=df_combined,
        x='Column',
        y='Value',
        hue='DF',
        ax=ax)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    return ax
