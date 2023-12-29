"""


"""
# %%

import numpy as np
import matplotlib.pyplot as plt

# Generate a sinusoidal signal with noise
np.random.seed(42)
t = np.linspace(0, 4 * np.pi, 100)
Y = 2 * np.sin(t) + 1 + 0.5 * np.random.randn(len(t))

# Initialize weights and bias
w1, w2, bias = np.random.randn(3)

# Learning rate
learning_rate = 0.01

# Maximum number of epochs
max_epochs = 100

# List to store mean squared error at each iteration
mse_list = []

# Training using the Widrow-Hoff rule
for epoch in range(max_epochs):
    mse = 0

    for i in range(2, len(t)):
        # Input
        x1 = Y[i - 1]
        x2 = Y[i - 2]
        y_true = Y[i]

        # Calculate predicted output
        y_pred = w1 * x1 + w2 * x2 + bias

        # Calculate error
        error = y_true - y_pred

        # Update weights and bias using the Widrow-Hoff rule
        w1 += learning_rate * error * x1
        w2 += learning_rate * error * x2
        bias += learning_rate * error

        # Update mean squared error
        mse += error ** 2

    # Calculate mean squared error for the current epoch
    mse /= len(t) - 2
    mse_list.append(mse)

# Final prediction using learned weights
y_pred_final = np.zeros_like(Y)
for i in range(2, len(t)):
    y_pred_final[i] = w1 * Y[i - 1] + w2 * Y[i - 2] + bias

# Plot the real signal, prediction, and mean squared error trend
plt.plot(t, Y, label='Real Signal')
plt.plot(t, y_pred_final, color='red', label='Widrow-Hoff Prediction')
plt.plot(t, mse_list, color='green', label='Mean Squared Error')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.title('Signal Prediction using Widrow-Hoff Rule')
plt.show()


# %%
