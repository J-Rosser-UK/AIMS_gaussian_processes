import torch
import matplotlib.pyplot as plt
import numpy as np
from covariance_functions import SquaredExponential

# Assuming the class SquaredExponential is already defined

# Instantiate the kernel with specific parameters
l = 1.0  # length scale
sigma_f = 1.0  # vertical variation
kernel = SquaredExponential(l=l, sigma_f=sigma_f)

# Generate input points for X1 and X2
X = torch.linspace(-5, 5, 100).unsqueeze(1)  # 100 points in range [-5, 5]

# Compute the covariance matrix (kernel matrix)
cov_matrix = kernel(X, X).detach().numpy()

# Plot the kernel matrix
plt.imshow(cov_matrix, cmap='viridis', extent=(-5, 5, -5, 5))
plt.colorbar()
plt.title('Squared Exponential Kernel')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Alternatively, plot a slice of the kernel
X_test = torch.linspace(-5, 5, 100).unsqueeze(1)
K_slice = kernel(torch.tensor([[0.0]]), X_test).detach().numpy()

plt.plot(X_test.numpy(), K_slice[0], label='Kernel Slice (X1 = 0)')
plt.title('Slice of Squared Exponential Kernel')
plt.xlabel('X2')
plt.ylabel('Covariance')
plt.legend()
plt.show()
