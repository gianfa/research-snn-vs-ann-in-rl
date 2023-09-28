""" Perceptron and delta rule """
# %%
import torch

# Define the training data
X = torch.tensor([
    [0., 0.], [0., 1.], [1., 0.], [1., 1.]]).reshape(4, 2, 1)
y = torch.tensor([0., 0., 0., 1.]).reshape(4, 1, 1)

# Initialize weights and bias
w = torch.rand(2).reshape(2, 1)
b = torch.rand(1).reshape(1, 1)

# Set learning rate and number of epochs
lr = 0.1
epochs = 300

# # Training
for epoch in range(epochs):
    error_sum = 0
    for i in range(len(X)):
        # Calculate the predicted output
        predicted_output = torch.matmul(X[i].T, w) + b
        
        # Compute the error
        error = y[i] - predicted_output
        error_sum += error
        
        # Update weights and bias using the delta rule
        w += lr * error * X[i]
        b += lr * error
    
    mse = error_sum / len(X)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Mean Squared Error = {mse}")

# Print the final weights and bias
print(f"Final Weights: {w}")
print(f"Final Bias: {b}")

# # Testing the perceptron

test_set = [
    [1., 1.],
    [0., 1.]
]
for x_test_i in test_set:
    x_test_i = torch.tensor([x_test_i]).reshape(2, 1)
    y_pred = torch.matmul(x_test_i.T, w) + b
    print((
        f"Test Input: {x_test_i.squeeze()}; "
        + f"pred probability: {y_pred.item():.2f}; "
        + f"pred output: {int(y_pred.item() > 0.5)}"))

# %%
