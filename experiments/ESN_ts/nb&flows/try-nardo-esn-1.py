# %%
import sys
sys.path += [
    "/Users/giana/Desktop/Github/Projects/research-snn-vs-ann-in-rl/experiments/ESN_ts/pytorch-esn-demolition/torchesn/utils/utilities.py",
    "/Users/giana/Desktop/Github/Projects/research-snn-vs-ann-in-rl/experiments/ESN_ts/pytorch-esn-demolition/torchesn/utils/",
    "/Users/giana/Desktop/Github/Projects/research-snn-vs-ann-in-rl/experiments/ESN_ts/pytorch-esn-demolition/torchesn/",
    "/Users/giana/Desktop/Github/Projects/research-snn-vs-ann-in-rl/experiments/ESN_ts/pytorch-esn-demolition/"
    ]
import utils
import numpy as np


ds_path = "/Users/giana/Desktop/Github/Projects/research-snn-vs-ann-in-rl/experiments/ESN_ts/pytorch-esn-demolition/examples/datasets/mg17.csv"
data = np.loadtxt(ds_path, delimiter=',', dtype=np.float32)



# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import torch.nn

from torchesn.nn import ESN
from torchesn import utils

# %%

data = np.loadtxt(ds_path, delimiter=',', dtype=np.float32)


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

output, hidden = model(input=trX, washout=washout, h_0=None, target=trY_flat)

model.fit()
output, hidden = model(trX, washout)
print("Training error:", loss_fcn(output, trY[washout[0]:]).item())

# Test
output, hidden = model(tsX, [0], hidden)
print("Test error:", loss_fcn(output, tsY).item())
print("Ended in", time.time() - start, "seconds.")

plt.plot(tsY.flatten(), label='Y')
plt.plot(output.flatten().detach().numpy(), label='Y_pred')
plt.legend()
# %%
