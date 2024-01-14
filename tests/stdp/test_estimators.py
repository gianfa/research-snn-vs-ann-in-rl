""" STDP Tests

$ python -m pytest tests/stdp/test_estimators.py -vv --pdb -s
"""
import collections
import sys
sys.path.append("../")
from typing import Callable, Iterable, List, Tuple  # noqa

import matplotlib.pyplot as plt
import pytest  # noqa
from sklearn.datasets import make_regression
import torch  # noqa
from torch import nn
from torch.optim.optimizer import Optimizer

from stdp.estimators import define_spiking_cluster
from stdp.funx import (stdp_generate_dw_lookup, stdp_step, 
    model_get_named_layer_params, model_layers_to_weights,
    )  # noqa
from stdp.spike_collectors import nearest_pre_post_pair
from stdp.estimators import ESN, BaseESN

@pytest.fixture
def simple_pre_post_W_A():

    W = torch.tensor([
        [0, 0.5],
        [0.5, 0],
    ])
    """Weight matrix"""

    A = torch.tensor([
        [0, 1],
        [-1, 0],
    ])
    """Adjacency matrix"""

    return W, A


@pytest.fixture
def dw_time_lookup_40():
    return stdp_generate_dw_lookup(40)


def test_BaseESN_simple__many_sizes():
    for _ in range(10):
        t_steps = torch.randint(12, 100, (1, 1)).item()
        X = torch.rand(2, t_steps)  # [=] (Nu, t)
        y = (X.sum(dim=0) > 1).int()[None, :]  # [=] (Ny, t)

        input_size = X.shape[0]
        reservoir_size = torch.randint(1, 100, (1, 1)).item()
        output_size = y.shape[0]
        spectral_radius = 0.9
        washout = 5

        esn = BaseESN(
            input_size=input_size,
            reservoir_size=reservoir_size,
            output_size=output_size,
            spectral_radius=spectral_radius,
            washout=washout
        )

        out = esn.train(X, y)
        assert out.shape == torch.Size([reservoir_size, t_steps - washout])

        predictions = esn.predict(X)
        assert predictions.shape == torch.Size(
            [output_size, t_steps - washout])


def test_define_spiking_cluster():
    clust = define_spiking_cluster(
        tot_neurons = 100,
    )
    weights = clust['weights']
    in_ns = clust['input']
    out_ns = clust['output']
    
    assert weights.ndim == 2
    assert weights.shape[0] == weights.shape[1]
    assert type(in_ns) == type(out_ns) == list


def test_ESN_simple():
    input_size = 1
    hidden_size = 100
    output_size = 1
    spectral_radius = 0.9

    esn = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        spectral_radius=spectral_radius)

    X = torch.linspace(-5, 5, 1000).unsqueeze(1)
    out = esn(X)
    assert out.shape == torch.Size([output_size, 1])


def test_ESN____init__spectral_radius():
    input_size = 1
    hidden_size = 100
    output_size = 1

    for sr in [0.2, 0.5, 2]:

        esn = ESN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            spectral_radius=sr)

        assert torch.linalg.eig(
            esn.W.weight).eigenvalues.abs().max().item() - sr < 1e4


def gen_int(n: int = 1, max_int: int = 100):
    return torch.randint(1, max_int, (1, n)).flatten().tolist()


def test_ESN_simple__many_sizes():
    for _ in range(10):
        input_size = gen_int(1)[0]
        hidden_size = gen_int(1)[0]
        output_size = gen_int(1)[0]
        spectral_radius = 0.9

        esn = ESN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            spectral_radius=spectral_radius)

        X = torch.linspace(-5, 5, input_size).reshape(1, input_size)
        out = esn(X)
        assert out.shape == torch.Size([output_size, 1])


def test_ESN_with_hidden_weights():
    input_size = 1
    hidden_size = 100
    output_size = 1
    spectral_radius = 0.9
    hidden_weights = torch.rand(hidden_size, hidden_size)
    
    # set zero positions to check later
    hidden_zeros_pos = [(0,4),(0,6)]
    hidden_weights[hidden_zeros_pos] = 0

    esn = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        hidden_weights=hidden_weights,
        spectral_radius=spectral_radius)

    input = torch.randn(1000, 1)
    out = esn(input)
    assert out.shape == torch.Size([output_size, 1])

    # now check for the zero positions
    hidden_W = [param for name, param in esn.named_parameters() if name == 'W.weight'][0]
    hidden_W_zeros = torch.argwhere(hidden_W==0)
    assert torch.allclose(hidden_W_zeros.T, torch.tensor(hidden_zeros_pos))


def test_ESN_simple__trainability():
    def train(
        model: nn.Module,
        X_y: List[Tuple[torch.TensorType, int]],
        optimizer: Optimizer,
        criterion: Callable = nn.BCEWithLogitsLoss(),
        epochs: int = 10) -> tuple:

        hist = {'loss': [], 'out': []}
        for epoch in range(epochs):
            for X, y in X_y:
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out.float(), y.unsqueeze(1).float())
                optimizer.step()
                hist['loss'].append(loss)
                hist['out'].append(out)
        return model, hist
    
    input_size = 1
    hidden_size = 100
    output_size = 1
    spectral_radius = 0.9

    esn = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        spectral_radius=spectral_radius)
    
    optimizer = torch.optim.Adam(esn.parameters())
    
    epochs = 20
    n_examples_per_class = 20
    X_y = (
        [(
            amp_i * torch.sin(torch.arange(100))[:, None],
            torch.tensor([0], dtype=torch.int8))
                for amp_i in range(n_examples_per_class)] +
            [(
                amp_i * torch.cos(torch.arange(100))[:, None],
                torch.tensor([1], dtype=torch.int8))
                    for amp_i in range(n_examples_per_class)]
    )
    X_y = [X_y[i] for i in torch.randperm(len(X_y)).tolist()]
    esn, hist = train(model=esn, X_y=X_y, optimizer=optimizer, epochs=epochs)

    assert len(hist['loss']) == epochs * len(X_y)

    # import pandas as pd
    # aa=pd.DataFrame(torch.tensor(hist['loss']))
    # aa.describe()

    #assert out.shape == torch.Size([output_size])


def test_ESN__train():

    # # Data Generation
    x, y, sk_coeffs = make_regression(
        n_samples=50,
        n_features=5,
        n_informative=5,
        n_targets=1,
        noise=5,
        coef=True,
        random_state=1
    )
    X = torch.tensor(x)
    Y = torch.tensor(y).unsqueeze(1)

    # # Define ESN
    input_size = X.shape[1]
    hidden_size = 100
    output_size = Y.shape[1]
    spectral_radius = 0.9

    esn = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        spectral_radius=spectral_radius)
    esn.train(X, Y, v=True)

    # Changing training `lr``
    esn = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        spectral_radius=spectral_radius)
    esn.train(X, Y, lr=4e0, v=True)


def test_ESN__train_multidim():

    # # Data Generation
    x, y, sk_coeffs = make_regression(
        n_samples=50,
        n_features=5,
        n_informative=5,
        n_targets=3,
        noise=5,
        coef=True,
        random_state=1
    )
    X = torch.tensor(x)
    Y = torch.tensor(y).unsqueeze(1)

    # # Define ESN
    input_size = X.shape[1]
    hidden_size = 100
    output_size = Y.shape[0]
    spectral_radius = 0.9

    # Changing training `lr``
    esn = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        spectral_radius=spectral_radius)
    esn.train(X, Y, lr=4e0, v=True)


def test_ESN__train_multibatch():

    # # Data Generation
    x, y, sk_coeffs = make_regression(
        n_samples=200,
        n_features=5,
        n_informative=5,
        n_targets=3,
        noise=5,
        coef=True,
        random_state=1
    )
    X = torch.tensor(x)
    Y = torch.tensor(y)

    bs = 50
    X = X.reshape((X.shape[0]//bs, bs, X.shape[1]))
    Y = Y.reshape((Y.shape[0]//bs, bs, Y.shape[1]))

    # # Define ESN
    input_size = X.shape[2]
    hidden_size = 100
    output_size = Y.shape[1]
    spectral_radius = 0.9

    # Changing training `lr``
    esn = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        spectral_radius=spectral_radius)
    esn.train(X, Y, lr=4e0, v=True)


def __SKIP__test_ESN__train__performance():
    from experimentkit.evaluation import regression # noqa

    # # Data Generation
    x, y, sk_coeffs = make_regression(
        n_samples=200,
        n_features=5,
        n_informative=5,
        n_targets=1,
        noise=5,
        coef=True,
        random_state=1
    )
    X = torch.tensor(x)  # [=] (200, 5)
    Y = torch.tensor(y).unsqueeze(1)  # [=] (200, 1)

    bs = 50
    X = X.reshape((X.shape[0]//bs, bs, X.shape[1]))  # [=] (4, 50, 5)
    Y = Y.reshape((Y.shape[0]//bs, bs, Y.shape[1]))  # [=] (4, 50, 1)

    # # Define ESN
    input_size = X.shape[2]
    hidden_size = 300
    output_size = Y.shape[1]
    spectral_radius = 0.9

    esn = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        spectral_radius=spectral_radius)
    esn.train(X, Y, epochs=400, v=True)

    bi = 0 # take a batch
    y_pred = esn(X[bi].to(torch.float32)).detach()

    report = regression.regression_report(Y[bi], y_pred)

    assert report['MAPE'] < 1

    # viz
    fig, ax = plt.subplots(sharey=True, sharex=True)
    ax.plot(Y[0].flatten() - y_pred.flatten())
    ax.set(
        title='$Y - y_pred$'
    )
    fig.tight_layout()
    fig.suptitle("Regression Plot", y=1.05)
    fig.savefig('my_plot.png')


def test_ESN__hidden_weights():

    # # Data Generation
    x, y, sk_coeffs = make_regression(
        n_samples=50,
        n_features=5,
        n_informative=5,
        n_targets=1,
        noise=5,
        coef=True,
        random_state=1
    )
    X = torch.tensor(x)
    Y = torch.tensor(y).unsqueeze(1)

    for hidden_size in torch.randint(1, int(1e3), (1, 8)).flatten().tolist():
        # # Define ESN
        input_size = X.shape[1]
        output_size = Y.shape[1]
        spectral_radius = 0.9
        hidden_weights=torch.eye(hidden_size)

        esn = ESN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            hidden_weights=hidden_weights,
            spectral_radius=spectral_radius)
        
        assert torch.nonzero(esn.W.weight).equal(torch.nonzero(hidden_weights))
    
    esn.train(X, Y, v=True)

    # python -m pytest tests/stdp/test_estimators.py -vv --pdb -s -k test_ESN__hidden_weights

