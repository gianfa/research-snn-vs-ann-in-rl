# %%
from itertools import combinations
import sys
from typing import Tuple
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np

from experimentkit_in.visualization import plot_n_examples
from experimentkit_in.generators.time_series import (
    ts_generate_constant, ts_generate_periodic_peak, ts_generate_periodic_sin,
    ts_add_onset, tsds_generate_periodic_sin_as_sum,
    tsds_generate_periodic_sin_as_prod_from_params
)

data_path = '../../data'

length = 100
baseline = 0

const = ts_generate_constant(length, baseline)
plt.plot(const)
plt.title('Constant')
plt.show()


pp = ts_generate_periodic_peak(length, max=5, step=10, baseline=baseline)
plt.plot(pp)
plt.title('Periodic Peak')
plt.show()


ps = ts_generate_periodic_sin(
    length, amplitude=10, phase=0.5, baseline=0, onset=0)
plt.plot(ps)
plt.title('Periodic Sin')
plt.show()


ps1 = ts_generate_periodic_sin(length, amplitude=10, phase=0.5)
ps2 = ts_generate_periodic_sin(length, amplitude=10, phase=1.8)
plt.plot(ps + ps2)
plt.title('Combination of Sin')
plt.show()

# %%

ps = ts_generate_periodic_sin(length, amplitude=10, phase=0.5)
ps_onset = ts_add_onset(200, ps, -1)
plt.plot(ps_onset)
plt.title('Offset')
plt.show()

# %% Example of combination of signals

length = 1000
baseline = -10

const = ts_generate_constant(length, baseline)
ps1 = np.concatenate((
        np.ones(200) * baseline,
        ts_generate_periodic_sin(length-200, amplitude=10, phase=0.5)
    ))
ps2 = ts_generate_periodic_sin(length, amplitude=10, phase=1.8)

plt.plot(ps1 + ps2)
plt.title('Combination of Sin 2')
plt.show()

# %% Dataset generation: Sum of Sins

X, y = tsds_generate_periodic_sin_as_sum(
    nexamples=57, length=987, ncats=10)
X.shape, y.shape

# %%

w1 = 1
w2 = w1 * 8e3

sig_2w = lambda t: (1 + np.cos(w1 * t) ) * (1 + np.sin(w2 * t))

# %% Dataset generation: Prod of Sin and Cos

length = 100

sig_2w = lambda w1, w2: (1 + np.cos(w1 * np.arange(length)) ) * (1 + np.sin(w2 * np.arange(100)))
w1 = 2
w2 = 23
w3 = 7
w4 = 91

a_12 = sig_2w(w1, w2)
a_34 = sig_2w(w3, w4)
sc1 = (a_12 * a_34)
plt.plot(sc1)

# %%


phases = [w1, w2]
amplitudes = [1, 1]
baselines = [1, 1]
X, y = tsds_generate_periodic_sin_as_prod_from_params(
    nexamples = 2,
    length=length,
    amplitudes=amplitudes,
    baselines=baselines,
    phases=phases,
)

print(X.shape, y.shape)
plot_n_examples(X, X.shape[0]);
# %%
