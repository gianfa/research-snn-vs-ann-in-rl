from functools import partial
from typing import List

from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

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