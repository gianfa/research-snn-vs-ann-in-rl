""" Visualize Neurons Caracteristics

* Lapique. Fixed C, changing beta

"""
# %% Trial1  Lapique

import sys

import matplotlib.pyplot as plt
import plotly.colors as colors
import plotly.subplots as sp
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import snntorch as snn
import torch

sys.path += ["..", "../..", "../../.."]

paramd = {
    'name': 'beta',
    'range': torch.linspace(0.1, 0.8, 10)
}


hist = {
    'mem_rec': [],
    'spk_rec': [],
    'cur_in': [],
}
for param in paramd['range']:
    print(param)
    neuron = snn.Lapicque(beta=param, C=5e-3, time_step=1e-3, threshold=0.5)

    # Initialize inputs and outputs
    cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.5), 0)
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec = [mem]
    spk_rec = [spk_out]

    # Neuron simulation
    for step in range(len(cur_in)):
        spk_out, mem = neuron(cur_in[step], mem)
        
        mem_rec.append(mem)
        spk_rec.append(spk_out)

    # convert lists to tensors
    hist['mem_rec'].append(torch.stack(mem_rec))
    hist['spk_rec'].append(torch.stack(spk_rec))
    hist['cur_in'].append(cur_in)


# %%
# Plot


fig = sp.make_subplots(rows=2, cols=1, row_heights=[0.25, 0.75])
t_steps = torch.arange(len(hist['cur_in'][0]))

cs = sample_colorscale('burg', torch.linspace(0, 1, 25).tolist())

fig.add_trace(
    go.Scatter(
        x=t_steps,
        y=hist['cur_in'][0].ravel(),
        name=f"{paramd['name']}",
    ),
    row=1, col=1
)

for i, param in enumerate(paramd['range']):
    
    fig.add_trace(
        go.Scatter(
            x=t_steps,
            y=hist['mem_rec'][i].ravel(),
            name=f"{param:.2f}",
            line={'color': cs[i]}
        ),
        row=2, col=1
    )

fig.update_layout(    
    {    
        'title': {
            'text': 'Lapique: beta<br>C=5e-3',
            'x': 0.5
        },
        # 'xaxis': {'title': 't steps'},
        'yaxis': {
            'title': 'Input current I<sub>in</sub>',
            'range': (-1, 1),
            'titlefont':{ 'size': 10 }
        },
        'xaxis2': {'title': 't steps'},
        'yaxis2': {
            'title': 'V membrane V<sub>mem</sub>',
            'range': (-1, 1),
            'titlefont':{ 'size': 10 }
        },
        'legend': {'title': paramd['name']}
    }
)

fig.show()

# %%
