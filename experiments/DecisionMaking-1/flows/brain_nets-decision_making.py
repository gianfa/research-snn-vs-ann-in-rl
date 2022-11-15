# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
""" [INCOMPLETE]
Title: examples of leaky neuron networks solving some of the exercises
ref1: Baldassarre 2020, https://drive.google.com/drive/folders/17JxiDyxWnZWj7tsXt6rkxCju9j99bz8K # noqa
ref2: https://www.istc.cnr.it/grouppage/locen-resources-initial-programming-exercises-simple-neural-networks # noqa
"""

from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
import math

# %%

# VARIABLES

# Simulation
sim_duration: float = 60.0  # [s]. Duration of the whole simulation
dt: float = 0.01  # [s]. Time step duration, to approximate continous t
sim_duration_steps: int = int(sim_duration/dt)

tau: Iterable[float] = np.array([0.2, 0.2])  # Speed of neurons' dynamics
n_units: int = len(tau)

# Input (data)
# Here two input signlas are defined
input: Iterable[float] = np.zeros((sim_duration_steps, n_units))

input[int(0.0/dt):int(5.0/dt), 0] = np.ones(int(5.0/dt)) * 0.0
input[int(0.0/dt):int(5.0/dt), 1] = np.ones(int(5.0/dt)) * 1.0

input[int(10.0/dt):int(15.0/dt), 0] = np.ones(int(5.0/dt)) * 0.1
input[int(10.0/dt):int(15.0/dt), 1] = np.ones(int(5.0/dt)) * 0.9

input[int(20.0/dt):int(25.0/dt), 0] = np.ones(int(5.0/dt)) * 0.2
input[int(20.0/dt):int(25.0/dt), 1] = np.ones(int(5.0/dt)) * 0.8

input[int(30.0/dt):int(35.0/dt), 0] = np.ones(int(5.0/dt)) * 0.3
input[int(30.0/dt):int(35.0/dt), 1] = np.ones(int(5.0/dt)) * 0.7

input[int(40.0/dt):int(45.0/dt), 0] = np.ones(int(5.0/dt)) * 0.4
input[int(40.0/dt):int(45.0/dt), 1] = np.ones(int(5.0/dt)) * 0.6

input[int(50.0/dt):int(55.0/dt), 0] = np.ones(int(5.0/dt)) * 0.5
input[int(50.0/dt):int(55.0/dt), 1] = np.ones(int(5.0/dt)) * 0.5

plt.plot(input)
plt.legend([0, 1])
plt.title("Input signals")
# %%

# Network
ap: Iterable[float] = np.zeros(n_units)  # action potential
ap_old: Iterable[float] = np.zeros(n_units)
activ: Iterable[float] = np.zeros(n_units)  # activation
activ_old: Iterable[float] = np.zeros(n_units)
w_leak: float = 0.2
w_late: float = 0.5

# Other variables

# # Data structures for data collection and analysis
mfHistActiPote = np.zeros((sim_duration_steps, n_units))
vfHistActi = np.zeros((sim_duration_steps, n_units))

# SIMULATION
for step in range(sim_duration_steps):
    ap_old = ap
    activ_old = activ

    ap[0] = ap_old[0] + \
        (dt/tau[0]) * (
          - w_leak * ap_old[0] + input[step, 0] - w_late * activ_old[1])
    ap[1] = ap_old[1] + \
        (dt/tau[1]) * (
          - w_leak * ap_old[1] + input[step, 1] - w_late * activ_old[0])

    activ[0] = max(0.0, (1.0 / ((1.0 + math.exp(-ap[0])))) * 2.0 - 1.0)
    activ[1] = max(0.0, (1.0 / ((1.0 + math.exp(-ap[1])))) * 2.0 - 1.0)

    # Data collection
    for unit_i in range(n_units):
        mfHistActiPote[step, unit_i] = ap[unit_i]
        vfHistActi[step, unit_i] = activ[unit_i]

# DATA ANALYSIS
plt.plot(input[:, 0], '--')
plt.plot(input[:, 1])
# plt.plot(mfHistActiPote[:,0],'o-')
# plt.plot(mfHistActiPote[:,1],'o-')
plt.plot(vfHistActi[:, 0], '--')
plt.plot(vfHistActi[:, 1])
# %%