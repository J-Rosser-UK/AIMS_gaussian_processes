import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt


class CovarianceFunction(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def plot(self, X_range=(-5, 5), num_points=100, slice_point=0.0):
        """
        Plots the covariance matrix and a 1D slice for any kernel.

        Parameters:
        X_range (tuple): The range of X values (min, max).
        num_points (int): Number of points to evaluate the kernel on.
        slice_point (float): The specific X1 value to plot the 1D slice for.
        """
        # Generate input points
        X = torch.linspace(X_range[0], X_range[1], num_points).unsqueeze(1)

        # Compute the covariance matrix using the kernel's forward method
        with torch.no_grad():
            cov_matrix = self.forward(X, X).numpy()

        # Plot the covariance matrix
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(cov_matrix, cmap='viridis', extent=(X_range[0], X_range[1], X_range[0], X_range[1]))
        plt.colorbar()
        plt.title('Covariance Matrix')
        plt.xlabel('X1')
        plt.ylabel('X2')

        # Compute a 1D slice for a fixed X1 value (e.g., X1 = slice_point)
        X1_slice = torch.tensor([[slice_point]])
        with torch.no_grad():
            slice_values = self.forward(X1_slice, X).numpy()

        # Plot the slice
        plt.subplot(1, 2, 2)
        plt.plot(X.numpy(), slice_values[0], label=f'Slice at X1 = {slice_point}')
        plt.title('Kernel Slice')
        plt.xlabel('X2')
        plt.ylabel('Covariance')
        plt.legend()

        plt.tight_layout()
        plt.show()
    

class SquaredExponential(CovarianceFunction):
    def __init__(self, l: float, sigma_f: float):
        super(SquaredExponential, self).__init__()
        # Register parameters so that they can be optimized
        self.l = nn.Parameter(torch.tensor(l))
        self.sigma_f = nn.Parameter(torch.tensor(sigma_f))
    
    def forward(self, X1, X2):
        dist = (X1.unsqueeze(1) - X2.unsqueeze(0)).pow(2).sum(2)
        return self.sigma_f**2 * torch.exp(-0.5 * dist / self.l**2)

class Periodic(CovarianceFunction):
    def __init__(self, l:float = 700.0, sigma_f:float = 1.0, p:float = 100.0):
        super(Periodic, self).__init__()
        self.l = nn.Parameter(torch.tensor(l))
        self.sigma_f = nn.Parameter(torch.tensor(sigma_f))
        self.p = nn.Parameter(torch.tensor(p))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        dist_matrix = (x1.unsqueeze(1) - x2.unsqueeze(0)).pow(2).sum(-1)
        return self.sigma_f**2 * torch.exp(-2 * torch.sin(np.pi * torch.sqrt(dist_matrix) / self.p)**2 / self.l**2)
    

class SquaredExponentialPlusPeriodic(CovarianceFunction):
    def __init__(self, l:float = 700.0, sigma_f:float = 1.0, p:float = 100.0):
        super(SquaredExponentialPlusPeriodic, self).__init__()
        self.l = nn.Parameter(torch.tensor(l))
        self.sigma_f = nn.Parameter(torch.tensor(sigma_f))
        self.p = nn.Parameter(torch.tensor(p))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        dist_matrix = (x1.unsqueeze(1) - x2.unsqueeze(0)).pow(2).sum(-1)
        return self.sigma_f**2 * torch.exp(-0.5 * dist_matrix / self.l**2) + self.sigma_f**2 * torch.exp(-2 * torch.sin(np.pi * torch.sqrt(dist_matrix) / self.p)**2 / self.l**2)


class SquaredExponentialTimesPeriodic(CovarianceFunction):
    def __init__(self, l:float = 700.0, sigma_f:float = 1.0, p:float = 100.0):
        super(SquaredExponentialTimesPeriodic, self).__init__()
        self.l = nn.Parameter(torch.tensor(l))
        self.sigma_f = nn.Parameter(torch.tensor(sigma_f))
        self.p = nn.Parameter(torch.tensor(p))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        dist_matrix = (x1.unsqueeze(1) - x2.unsqueeze(0)).pow(2).sum(-1)
        return self.sigma_f**2 * torch.exp(-0.5 * dist_matrix / self.l**2) * self.sigma_f**2 * torch.exp(-2 * torch.sin(np.pi * torch.sqrt(dist_matrix) / self.p)**2 / self.l**2)
    

if __name__ == "__main__":
    # Plot all covariance functions

    # Squared Exponential
    kernel = SquaredExponential(l=50.0, sigma_f=1.0)
    kernel.plot(X_range=(-1000, 1000), num_points=100, slice_point=0.0)

    # Periodic
    kernel = Periodic(l=50.0, sigma_f=1.0, p=500.0)
    kernel.plot(X_range=(-1000, 1000), num_points=100, slice_point=0.0)

    # Squared Exponential + Periodic
    kernel = SquaredExponentialPlusPeriodic(l=50.0, sigma_f=1.0, p=20.0)
    kernel.plot(X_range=(-1000, 1000), num_points=100, slice_point=0.0)

    # Squared Exponential * Periodic
    kernel = SquaredExponentialTimesPeriodic(l=50.0, sigma_f=1.0, p=20.0)
    kernel.plot(X_range=(-1000, 1000), num_points=100, slice_point=0.0)