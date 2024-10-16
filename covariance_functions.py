import torch
import numpy as np
import torch.nn as nn

class CovarianceFunction(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    

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
