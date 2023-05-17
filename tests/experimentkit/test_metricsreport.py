""" Tests for MetricsReport

python -m pytest tests/experimentkit/test_metricsreport.py -vv --pdb
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experimentkit_in.metricsreport import MetricsReport


def test_MetricsReport__init():
    mr = MetricsReport()
    assert type(mr) == MetricsReport


def test_MetricsReport__append():
    mr = MetricsReport()

    y_true = [1,0,1]
    y_pred = [1,1,0]
    mr.append(y_true, y_pred)
    assert mr.y_pred[0] ==  y_pred
    assert mr.y_true[0] == y_true


    y_true2 = [1,1,1]
    y_pred2 = [0,0,0]
    mr.append(y_true2, y_pred2)
    assert mr.y_pred == [y_pred, y_pred2]
    assert mr.y_true == [y_true, y_true2]


def test_MetricsReport__get_true_pred():
    mr = MetricsReport()

    # Same shape appended data
    y_true = [1,0,1]
    y_pred = [1,1,0]
    mr.append(y_true, y_pred)
    y_true2 = [1,1,1]
    y_pred2 = [0,0,0]
    mr.append(y_true2, y_pred2)
    yt, yp = mr.get_true_pred()

    assert len(yt) == len(y_true) + len(y_true2)
    assert len(yp) == len(y_pred) + len(y_pred2)

    # Different shape appended data
    mr = MetricsReport()
    y_true = [1,0,1]
    y_pred = [1,1,0]
    mr.append(y_true, y_pred)
    y_true2 = [1,1]
    y_pred2 = [0,0]
    mr.append(y_true2, y_pred2)
    y_true3 = [0,0]
    y_pred3 = [1,1]
    mr.append(y_true3, y_pred3)
    yt, yp = mr.get_true_pred()

    assert len(yt) == sum([len(yi) for yi in [y_true, y_true2, y_true3]])
    assert len(yp) == sum([len(yi) for yi in [y_pred, y_pred2, y_pred3]])


def test_MetricsReport__get_metrics():
    mr = MetricsReport()

    # Same shape appended data
    y_true = [1,0,1]
    y_pred = [1,1,0]
    mr.append(y_true, y_pred)
    y_true2 = [1,1,1]
    y_pred2 = [0,0,0]
    mr.append(y_true2, y_pred2)
    metrics = mr.get_metrics()

    assert type(metrics) == np.ndarray


    metrics, metrics_names = mr.get_metrics(return_names=True)
    assert type(metrics) == np.ndarray
    assert metrics_names == ['accuracy', 'precision', 'f1', 'n_samples']

    metrics = mr.get_metrics(return_frame=True)
    assert type(metrics) == pd.DataFrame


