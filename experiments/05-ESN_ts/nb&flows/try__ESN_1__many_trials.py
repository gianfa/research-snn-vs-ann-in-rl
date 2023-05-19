# %%
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.sparse import random
from scipy.sparse.linalg import eigsh

import torch

class ESN:
    """
    Echo State Network
    """

    def __init__(self, Nr, **kwargs):
        """
        Constructor

        Args:
        - Nr: reservoir's size

        Optional Args:
        - leakRate: leakage rate
        - spectralRadius: spectral radius
        - inputScaling: input weights scale
        - biasScaling: bias weights scale
        - regularization: regularization parameter
        - connectivity: reservoir connectivity
        - readoutTraining: readout training method
        """

        # Set default values for hyperparameters
        self.alpha = 1
        self.rho = 0.9
        self.inputScaling = 1
        self.biasScaling = 1
        self.lmbd = 1
        self.connectivity = 1
        self.readout_training = 'ridgeregression'

        # Set the number of neurons in the reservoir
        self.Nr = Nr

        # Override default values if specified
        for key, value in kwargs.items():
            if key == 'leakRate':
                self.alpha = value
            elif key == 'spectralRadius':
                self.rho = value
            elif key == 'inputScaling':
                self.inputScaling = value
            elif key == 'biasScaling':
                self.biasScaling = value
            elif key == 'regularization':
                self.lmbd = value
            elif key == 'connectivity':
                self.connectivity = value
            elif key == 'readoutTraining':
                self.readout_training = value
            else:
                raise ValueError('the option does not exist')

    def readout_training(self, X, Y, esn):
        return np.linalg.pinv(X.T @ X + esn.ridgeParam * np.eye(1 + esn.Nr + X.shape[1])) @ X.T @ Y
    
    def train(self, trX: list, trY, washout):
        """
        Trains the network on input X given target Y.

        Args:
            trX: list of length N, where each element is an array of shape
                (sequenceLenght, sequenceDimension).
            trY: target matrix composed by all sequences.
                Washout must be applied before calling this function.
            washout: number of initial timesteps not to collect.
        """
        seqDim = trX[0].shape[1]
        N = len(trX)
        trainLen = trY.shape[0]
        
        self.Win = self.inputScaling * (np.random.rand(self.Nr, seqDim) * 2 - 1)
        self.Wb = self.biasScaling * (np.random.rand(self.Nr, 1) * 2 - 1)
        self.Wr = scipy.sparse.rand(self.Nr, self.Nr, density=self.connectivity).toarray()
        #self.Wr = np.random.rand(self.Nr, self.Nr, density=self.connectivity).toarray()
        self.Wr[self.Wr != 0] = self.Wr[self.Wr != 0] * 2 - 1
        self.Wr = self.Wr * (
            self.rho / np.max(np.abs(np.linalg.eig(self.Wr)[0])))
        
        X = torch.zeros((1 + seqDim + self.Nr, trainLen))
        idx = 0
        for ti in range(N):
            U = trX[ti].T
            x = torch.zeros((self.Nr, 1))
            for i in range(U.shape[1]):
                u = U[:, i:i+1]
                x_ = torch.tanh(
                    torch.mm(
                        torch.tensor(self.Win).float(),
                        torch.tensor(u).float()
                    ) + torch.mm(
                        torch.tensor(self.Wr).float(),
                        torch.tensor(x).float()
                        ) + self.Wb)
                x = (1 - self.alpha) * x + self.alpha * x_
                if i > washout:
                    X[:, idx:idx+1] = np.vstack((np.array([1]), u, x))
                    idx += 1
        self.internalState = X[1+seqDim:, :]
        # self.Wout = self.readout_training(X, trY)
        self.Wout = self.ridgeregression(X, trY.squeeze(1))
    
    def ridgeregression(self, X, Y):
        return torch.inverse(
            X.T @ X + self.lmbd * torch.eye(X.shape[0]) @ X.T @ Y
        )

    def predict(self, data, washout):
        """
        Computes the output given the data.

        Args:
        - self: an instance of EchoStateNetwork class
        - data: list of numpy arrays of shape (sequence_length, sequence_dimension) representing the input data
        - washout: number of initial timesteps to not collect

        Returns:
        - y: predicted output
        """

        seq_dim = data[0].shape[1]
        N = len(data)
        train_len = 0
        for s in range(N):
            train_len += data[s].shape[0] - washout

        X = torch.zeros(1 + seq_dim + self.Nr, train_len, device=self.device)
        idx = 0
        for s in range(N):
            U = data[s].T
            x = torch.zeros((self.Nr, 1), device=self.device)

            for i in range(U.shape[1]):
                u = U[:, i].reshape(-1, 1)
                x_ = torch.tanh(
                    torch.mm(self.Win, u) + torch.mm(self.Wr, x) + self.Wb)
                x = (1 - self.alpha) * x + self.alpha * x_
                if i >= washout:
                    X[:, idx] = torch.cat(
                        [
                            torch.tensor([1],
                            device=self.device), u.flatten(), x.flatten()
                        ])
                    idx += 1

        esn.internalState = X[1 + seq_dim : , :]
        y = torch.mm(esn.Wout, X)
        y = y.T

        return y.detach().cpu().numpy()
               


esn = ESN(Nr = 50)
# %%
# Data creation
n_examples_per_class = 60 
X_y = (
    [(
        amp_i * torch.sin(torch.arange(100))[:, None],
        torch.tensor([0], dtype=torch.int8))
            for amp_i in range(n_examples_per_class)] +
        [(
            amp_i * (torch.cos(torch.arange(100)) * torch.sin(torch.arange(100)))[:, None],
            torch.tensor([1], dtype=torch.int8))
                for amp_i in range(n_examples_per_class)]
)
# X_y [=] X_i, y_i; X_i [=] N x 1; y_i [=] 1
X_y = [X_y[i] for i in torch.randperm(len(X_y)).tolist()]
train_len = int(len(X_y) * 0.8)
print(f"train length: {train_len}")
X_y_train = X_y[:train_len]
X_y_test = X_y[train_len:]

examples = np.vstack([X_y[i][0].numpy().T for i in range(6)])

# %%
import sys
sys.path += [
    "./pytorch-esn/torchesn/utils/utilities.py",
    "./pytorch-esn/torchesn/utils/",
    "./pytorch-esn/torchesn/"
    ]
import utils

ds_path = "./pytorch-esn/examples/datasets/mg17.csv"
data = np.loadtxt(ds_path, delimiter=',', dtype=np.float32)

X_data = np.expand_dims(data[:, [0]], axis=1)
Y_data = np.expand_dims(data[:, [1]], axis=1)
X_data = torch.from_numpy(X_data)
Y_data = torch.from_numpy(Y_data)

trX = X_data[:5000] # [=] [5000, 1, 1]
trY = Y_data[:5000] # [=] [5000, 1, 1]
tsX = X_data[5000:]
tsY = Y_data[5000:]
plt.plot(trX.flatten())


washout = [500]
trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)

input_size = output_size = 1
hidden_size = 500
loss_fcn = torch.nn.MSELoss()

# %%

model = ESN(Nr=(5,5))

esn.train(trX, trY, washout=50)
Y_pred = esn.predict(tsX)
# %%
