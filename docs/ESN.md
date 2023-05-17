# Echo State Network




## What happen during a training

Let's say we have the following setup.  
Data  
* `X`: 200 x 20 sequence tensor, namely made by 20 features over 200 time steps.
* `Y`: 200 x 1 sequence tensor, namely made by 1 feature over 200 time steps.

### Define the newtork size
```python
# ESN definition
input_size = 20  # the number of input features (X cols)
hidden_size = 100  # an arbitrary number chosen according to your hypothesis
output_size = 1 # the number of output features (Y cols)
```

## Training high-level

The ESN processes time step by time step the tensor.
The training is performed batch-wise.

```
# X [=] (n_batches, bs, in_features); # bs: batch size = time length
# Y [=] (n_batches, bs, out_features)

for epoch in epochs:
    for bi in n_batches:
        # Xi [=] (bs, in_features)
        # Yi [=] (bs, out_features)
        y_pred_i = esn(Xi)  # [=] (bs, out_features)
```

## Training low-level

```

```