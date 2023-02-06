# %%
""" Results Data Analysis

"""
import os
from pathlib import Path
import sys

import pandas as pd

sys.path.append("../..")
from experimentkit.funx import pickle_load

dir_path = "./data"
name_condition = lambda name: "hidden_" in name and "td_" in name

d = Path(dir_path)
candidates_path = [fi for fi in d.iterdir() if name_condition(fi.name)]
candidates = []
for ci in candidates_path:
    cid = pickle_load(ci)
    cid['n_hidden'] = int(
        str(ci).split('hidden_')[1].split("-")[0])
    candidates.append(cid)

fields_to_include = [
    'batch_size', 'num_epochs', 'net_inner_delta_t',
    'training-time', 'test-metrics', 'n_hidden'   
]
select_fields = lambda dd: {
    k:v for k, v in dd.items() if k in fields_to_include}
candidates = [select_fields(ci) for ci in candidates]

candidates_updated = []
for di in candidates:
    cu = di.copy()
    ti = di['test-metrics'].to_dict(orient='row')[0]
    cu.update(**ti)
    del cu ['test-metrics']
    candidates_updated.append(cu)
candidates = candidates_updated
candidates = pd.DataFrame(candidates)[[
    'n_hidden', 'net_inner_delta_t', 'batch_size', 'num_epochs',
    'training-time', 'accuracy', 'precision', 'f1', 'n_samples']]

candidates.plot.scatter(
  x='n_hidden', y='accuracy', color='net_inner_delta_t', colormap='YlOrRd')
candidates.plot.scatter(
  x='net_inner_delta_t', y='accuracy', color='n_hidden', colormap='YlOrRd')


candidates
# %%
