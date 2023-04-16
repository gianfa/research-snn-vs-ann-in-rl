# %%
""" Oja's Rule



Decription
----------


References
----------
1. `Erkki Oja (2008) Oja learning rule. Scholarpedia, 3(3):3612 <http://www.scholarpedia.org/article/Oja_learning_rule>`_.
2. `Oja's Rule implementation, neuronaldynamics-exercises<https://neuronaldynamics-exercises.readthedocs.io/en/latest/_modules/neurodynex3/ojas_rule/oja.html#learn>`_.
3. `YUSOFF, Mohd-Hanif; CHROL-CANNON, Joseph; JIN, Yaochu. Modeling neural plasticity in echo state networks for classification and regression. Information Sciences, 2016, 364: 184-196. <>`_.

Theory
------
Oja's Rule improves Hebbian Rule adding a "forgetting" term proportional to 
both the weight and the output of the neuron.[R.1]
It can easily lead to PCA or ICA computation.

"""


# %% Imports

import sys
sys.path.append("..")

import time
import os
from daft import PGM
from typing import Dict, List
import seaborn as sns
import snntorch as snn  # noqa
from snntorch import spikeplot as splt  # noqa
from snntorch import spikegen  # noqa
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix  # noqa

import torch  # noqa
import torch.nn as nn  # noqa
from torch.utils.data import DataLoader  # noqa
from torchvision import datasets, transforms  # noqa

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd # noqa
import itertools  # noqa

from experimentkit_in.metricsreport import MetricsReport

# %%

A = np.array([
    [0, 1, 0],
    [-1, 0, -1],
    [0, 1, 0],
])

W = np.array([
    [0, 0.3, 0],
    [0, 0,   0],
    [0, 0.3, 0],
])

# %%

#test
W = np.random.rand(10, 10)

mat = W
mat_names = {}

cols = 3
rows = np.ceil(mat.shape[0]/4)
pgm = PGM(shape=(rows, cols))
# assumes W is a all to all matrix
for ni in range(mat.shape[0]):
    ri = int(np.floor(ni/cols))
    ci = int(ni % cols)
    pgm.add_node(daft.Node(f"N_{ni}", f"{ni}", ci, ri))
    mat_names[ci, ri] = f"N_{ni}"

# for pre in range(W.shape[0]):
#     for post in range(W.shape[1]):
#         pgm.add_edge('A', 'D')

pgm.render()
plt.show()
# %%
# C: confounder causing spurious corr A-R
# pgm.add_node(daft.Node('C', r"C", 1, 1))


# pgm.add_node(daft.Node('D', r"D", 1, 2))
# pgm.add_node(daft.Node('A', r"A", 2, 1))
# pgm.add_node(daft.Node('S', r"Sick", 3, 2))
# pgm.add_node(daft.Node('R', r"R", 2, 3))
# pgm.add_node(daft.Node('I', r"I", 2, 2))
# pgm.add_node(daft.Node('P', r"P", 1, 1))

# # pgm.add_edge('C', 'A')
# # # pgm.add_edge('C', 'R')

# pgm.add_edge('A', 'I')
# pgm.add_edge('A', 'D')
# pgm.add_edge('I', 'R')
# pgm.add_edge('S', 'R')
# pgm.add_edge('S', 'A')

# # pgm.add_edge('D', 'C')
# pgm.add_edge('D', 'R')
# pgm.add_edge('P', 'D')
# pgm.add_edge('A', 'P')

pgm.render()
plt.show()

# %%


        