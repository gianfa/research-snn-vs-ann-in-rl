# %%
import sys
sys.path += ["..", "../..", "../../.."]
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset  # noqa
import numpy as np
from itertools import combinations
from experimentkit_in.visualization import plot_n_examples
from typing import List, Tuple

batch_size: int = 32

# List the phases
w1 = 2
w2 = w1 * 3
w3 = 4
w4 = w3 * 3

sig_length = 40 # length of the Xi signals
tot_sig_length = 20 * 40 # length of the parent sig, to be cut to have many Xi



dtype = torch.float
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


# Definition of the signal function
sig_2w = lambda w1, w2, t: (1 + np.cos(w1 * t) ) * (1 + np.sin(w2 * t))

ws = [w1, w2, w3, w4]

def ts_get_biphase_signals_combinations_wt_labels(
        ws: List[float],
        signal_function: callable,
        signal_len: int,
        encode_labels: bool = True
    ):
    t = np.arange(signal_len)
    
    ws_combinations = list(combinations(labels, 2))
    all_sig_combs = [
        signal_function(w_1, w_2, t) for w_1, w_2 in ws_combinations]
    
    # create integer labels
    all_sig_labels_combs = list(combinations(np.arange(len(ws)), 2))
    
    if not encode_labels:
        return all_sig_combs, all_sig_labels_combs
    
    ncats = len(ws_combinations)
    int2oh = lambda i, ncats: np.identity(ncats)[i, :]
    oh2int = lambda oh: np.argwhere(oh).flatten()

    labels = ws_combinations
    label2oh_dict = {ws_combinations[i]:int2oh(i, ncats) for i in range(ncats)}
    
    # define dict to encode and decode
    label2oh = lambda label: label2oh_dict[label]
    oh2label = lambda oh: ws_combinations[oh2int(oh)]

    






parent_signals, all_sig_labels_combs = \
    ts_get_biphase_signals_combinations_wt_labels(
        ws, sig_2w, tot_sig_length)


plot_n_examples(parent_signals, 6)


# Label Encoding
labels = np.arange(len([w1, w2, w3, w4]))

ws_combinations = list(combinations(labels, 2))
ncats = len(ws_combinations)

int2oh = lambda i, ncats: np.identity(ncats)[i, :]
oh2int = lambda oh: np.argwhere(oh).flatten()

labels = ws_combinations
label2oh_dict = {ws_combinations[i]:int2oh(i, ncats) for i in range(ncats)}
label2oh = lambda label: label2oh_dict[label]
oh2label = lambda oh: ws_combinations[oh2int(oh)]

# int2label = {i:ws_combinations[i] for i in range(ncats)}
# label2int = {wsi: i for i, wsi in int2label.items()}


# %% Dataset Creation: batching

def ts_get_signals_and_labels(
        ws: list,  # the original signals
        labels: list,
        sig_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    
    It discards the last batch if its length is lower than sig_length

    Examples
    --------
    >>> nsignals = 3
    >>> signals = np.random.rand(100, parent_signals)
    >>> labels = np.arange(nsignals)
    >>> signals, signals_labels = ts_get_signals_and_labels(
    ...     signals, labels, 32)
    """

    labels = np.arange(len(ws))

    # get all the parent signals combinations
    parent_signals = [sig_2w(w_1, w_2, t) for w_1, w_2 in combinations(ws, 2)]

    ws_combinations = list(combinations(labels, 2))
    ncats = len(ws_combinations)

    int2oh = lambda i, ncats: np.identity(ncats)[i, :]
    oh2int = lambda oh: np.argwhere(oh).flatten()

    labels = ws_combinations
    label2oh_dict = {ws_combinations[i]:int2oh(i, ncats) for i in range(ncats)}
    label2oh = lambda label: label2oh_dict[label]
    oh2label = lambda oh: ws_combinations[oh2int(oh)]

    signals = []
    signals_labels = []
    for i, psi_li in enumerate(zip(parent_signals, labels)):
        psi, li = psi_li # parent_signal_i, label_i

        # fullfill the batches
        sig_i_b = [] # [batch_0_s1 batch_1_s1, ..., batch_n_s1]
        for i in range(0, len(psi), sig_length):
            sig_i_b_i = psi[i:min(i + sig_length, len(psi))]
            if len(sig_i_b_i) == sig_length:
                sig_i_b.append(sig_i_b_i)
        
        li_i = [label2oh(li) for _ in sig_i_b]
        signals.append(sig_i_b)
        signals_labels.append(li_i)
    assert len(signals) == len(signals) == len(signals_labels)

    signals = torch.Tensor(np.vstack(signals)).float()
    signals_labels = torch.Tensor(np.vstack(signals_labels)).int()
    assert len(signals) == len(signals_labels)
    return signals, signals_labels




signals, signals_labels = ts_get_signals_and_labels(
    parent_signals, labels, sig_length=sig_length)

plot_n_examples(signals, 6)
print(
    f"signals.shape: {signals.shape},\n"+
    f"signals_labels.shape: {signals_labels.shape}")
# %% Dataset Creation: load into DataLoaders

class BiPhasicDataset(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y)
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.X)