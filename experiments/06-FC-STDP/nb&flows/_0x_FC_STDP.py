# %%

from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path += ["..", "../..", "../../.."]
from stdp.funx import stdp_step

ROOT_DIR = Path("../../../")

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)   #
        self.fc2 = nn.Linear(5, 3)    #
        self.fc3 = nn.Linear(3, 2)    #

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = SimpleNet()

input_tensor = torch.randn(1, 10)

output = net(input_tensor)

print(output)
# %%

bs = 10
dt_stdp = 5
th = 0.5
train_batch = torch.randn(bs, 10)

for i, data_targets in enumerate(train_batch):
    out = net(input_tensor)
    spks = out > th

    if i > 0 and i % dt_stdp == 0:
        steps = (i - dt_stdp, i)
        # cur_raster = raster[steps[0]:steps[1]]

        cur_raster = spk_out[steps[0]:steps[1]]
        weights = dict(net.named_parameters())['fc2.weight']
        new_weights=stdp_step(
            weights=weights,
            connections=None,
            raster=cur_raster,
            # spike_collection_rule=all_to_all,
            # dw_rule="sum",
            bidirectional=True,
            max_delta_t=20,
            inplace=False,
            v=False,
        )
        for name, param in net.named_parameters():
            if name == "fc2.weight":
                param.data = nn.parameter.Parameter(new_weights.clone())
        assert dict(net.named_parameters())['fc2.weight'] == new_weights
        # store the new weights
        weights['fc2'].append(new_weights.clone())
        stdp_dt_idxs.append(step)