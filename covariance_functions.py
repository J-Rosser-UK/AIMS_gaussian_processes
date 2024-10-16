import torch
import numpy as np

class CovarianceFunction:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError
    

class SquaredExponential(CovarianceFunction):
    def __init__(self, l:float = 700.0, sigma_f:float = 1.0):
        self.l = l
        self.sigma_f = sigma_f

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor):
      
        dist_matrix = (x1.unsqueeze(1) - x2.unsqueeze(0)).pow(2).sum(-1)

        return self.sigma_f**2 * torch.exp(-0.5 * dist_matrix / self.l**2)
    

class Periodic(CovarianceFunction):
    def __init__(self, l:float = 700.0, sigma_f:float = 1.0, p:float = 100.0):
        self.l = l
        self.sigma_f = sigma_f
        self.p = p

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor):
        dist_matrix = (x1.unsqueeze(1) - x2.unsqueeze(0)).pow(2).sum(-1)
        return self.sigma_f**2 * torch.exp(-2 * torch.sin(np.pi * torch.sqrt(dist_matrix) / self.p)**2 / self.l**2)
    

class SquaredExponentialPlusPeriodic(CovarianceFunction):
    def __init__(self, l:float = 700.0, sigma_f:float = 1.0, p:float = 100.0):
        self.l = l
        self.sigma_f = sigma_f
        self.p = p

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor):
        dist_matrix = (x1.unsqueeze(1) - x2.unsqueeze(0)).pow(2).sum(-1)
        return self.sigma_f**2 * torch.exp(-0.5 * dist_matrix / self.l**2) + self.sigma_f**2 * torch.exp(-2 * torch.sin(np.pi * torch.sqrt(dist_matrix) / self.p)**2 / self.l**2)
