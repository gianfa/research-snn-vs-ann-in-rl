# %%
import sys
sys.path += [
    "/Users/giana/Desktop/Github/Projects/research-snn-vs-ann-in-rl/experiments/ESN_ts/pytorch-esn/torchesn/utils/utilities.py",
    "/Users/giana/Desktop/Github/Projects/research-snn-vs-ann-in-rl/experiments/ESN_ts/pytorch-esn/torchesn/utils/",
    "/Users/giana/Desktop/Github/Projects/research-snn-vs-ann-in-rl/experiments/ESN_ts/pytorch-esn/torchesn/",
    "/Users/giana/Desktop/Github/Projects/research-snn-vs-ann-in-rl/experiments/ESN_ts/pytorch-esn/torchesn",
    "/Users/giana/Desktop/Github/Projects/research-snn-vs-ann-in-rl/experiments/ESN_ts/pytorch-esn"
    ]
ds_path = "/Users/giana/Desktop/Github/Projects/research-snn-vs-ann-in-rl/experiments/ESN_ts/pytorch-esn/examples/datasets/mg17.csv"

# %%
import torch.nn
import numpy as np
from torchesn.nn import ESN
from torchesn import utils
import time

# %%

data = np.loadtxt(ds_path, delimiter=',', dtype=np.float32)



dtype = torch.double
torch.set_default_dtype(dtype)

if dtype == torch.double:
    data = np.loadtxt('datasets/mg17.csv', delimiter=',', dtype=np.float64)
elif dtype == torch.float:
    data = np.loadtxt('datasets/mg17.csv', delimiter=',', dtype=np.float32)
X_data = np.expand_dims(data[:, [0]], axis=1)
Y_data = np.expand_dims(data[:, [1]], axis=1)
X_data = torch.from_numpy(X_data)
Y_data = torch.from_numpy(Y_data)

trX = X_data[:5000] # [=] [5000, 1, 1]
trY = Y_data[:5000] # [=] [5000, 1, 1]
tsX = X_data[5000:]
tsY = Y_data[5000:]

washout = [500]
input_size = output_size = 1
hidden_size = 500
loss_fcn = torch.nn.MSELoss()

# %%

start = time.time()

# Training
trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)

model = ESN(input_size, hidden_size, output_size)

model(trX, washout, None, trY_flat)
model.fit()
output, hidden = model(trX, washout)
print("Training error:", loss_fcn(output, trY[washout[0]:]).item())

# Test
output, hidden = model(tsX, [0], hidden)
print("Test error:", loss_fcn(output, tsY).item())
print("Ended in", time.time() - start, "seconds.")

# %%
