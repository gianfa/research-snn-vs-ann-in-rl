""" Just a sketch book

"""

# %%import sys
import sys

sys.path += ["..", "../..", "../../.."]
from experimentkit_in.generators.time_series import gen_lorenz

ds = gen_lorenz()

shift = 2
X = ds[:-shift]
y = ds[shift:, 0]

print(X.shape, y.shape)
assert X.shape[0] == y.shape[0]

# %%

import torch


def single_index_to_coordinate(index, num_columns):
    row = index // num_columns
    col = index % num_columns
    return row, col


def generate_sparse_matrix(shape, density, values_f=torch.rand):
    num_elements = int(shape[0] * shape[1] * density)
    indices = torch.randperm(shape[0] * shape[1])[:num_elements]
    indices = torch.Tensor([
        single_index_to_coordinate(i, shape[1]) for i in indices])
    values = values_f(num_elements)
    matrix = torch.sparse_coo_tensor(indices.T, values, torch.Size(shape))
    return matrix


# Esempio di utilizzo
shape = (5, 5)  # Dimensione desiderata della matrice
density = 0.3  # Densità desiderata (percentuale di elementi non nulli)

sparse_matrix = generate_sparse_matrix(shape, density)
print(sparse_matrix.to_dense())

# %%
