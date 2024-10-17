import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from load_data import get_data
from covariance_functions import CovarianceFunction, SquaredExponential, Periodic, SquaredExponentialPlusPeriodic, SquaredExponentialTimesPeriodic
from tqdm import tqdm
   

class GaussianProcess:
    def __init__(self, kernel: CovarianceFunction, noise_variance: float=0.2, jitter:float = 1e-8):   
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.jitter = jitter

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor):
        self.X_train = X_train
        self.y_train = y_train

        # Compute the covariance matrix of the training points
        K = self.kernel.forward(X_train, X_train)
        
        # Add noise to the diagonal
        noise_variance = self.noise_variance * torch.eye(X_train.size(0))
        K += noise_variance

        # Add jitter to the diagonal
        jitter = self.jitter * torch.eye(X_train.size(0))
        K += jitter

        print(np.linalg.cond(K.detach().numpy()))
        
        # Cholesky inverse
        L = torch.linalg.cholesky(K)
        self.K_inv = torch.cholesky_inverse(L)

    def predict(self, X_test: torch.Tensor):
        # Covariance between test points and training points
        K_star = self.kernel.forward(X_test, self.X_train)
        K_star_star = self.kernel.forward(X_test, X_test)
        
        # Compute the predictive mean
        mean_s = K_star @ self.K_inv @ self.y_train
        
        # Compute the predictive covariance
        cov_s = K_star_star - K_star @ self.K_inv @ K_star.T
        
        return mean_s, cov_s
    
    def log_marginal_likelihood(self):
        # Compute log marginal likelihood
        K = self.kernel.forward(self.X_train, self.X_train)
        noise_variance = self.noise_variance * torch.eye(self.X_train.size(0))
        K += noise_variance + self.jitter * torch.eye(self.X_train.size(0))
        
        L = torch.linalg.cholesky(K)
    
        alpha = torch.cholesky_solve(self.y_train, L)

        log_marg_likelihood = -0.5 * self.y_train.T @ alpha - torch.sum(torch.log(torch.diag(L))) - 0.5 * self.X_train.size(0) * torch.log(2 * torch.tensor(np.pi))

        return log_marg_likelihood
    
    def optimize_hyperparameters(self, learning_rate=0.01, n_iters=100):
        # Define the hyperparameters you want to optimize (e.g., kernel parameters)
        # Assume self.kernel has parameters l and sigma_f as tensors
        optimizer = torch.optim.Adam(self.kernel.parameters(), lr=learning_rate)

        for i in tqdm(range(n_iters)):
            optimizer.zero_grad()  # Clear the gradients
            loss = -self.log_marginal_likelihood()  # Minimize the negative log marginal likelihood
            loss.backward()  # Compute the gradients
            optimizer.step()  # Update the hyperparameters
            
            if i % 10 == 0:
                print(f"Iteration {i+1}/{n_iters} - Loss: {loss.item()}")

        # Print the final optimized hyperparameters
        print(f"Optimized hyperparameters: l={self.kernel.l.item()}, sigma_f={self.kernel.sigma_f.item()}")
    

def plot_gp(X_train, y_train, X_test, X_underlying, y_underlying, mean_pred, cov_pred):
    std = np.sqrt(np.diag(cov_pred.detach().numpy())) 
    mean_pred = mean_pred.detach().numpy()

   
    
    fig = plt.figure(figsize=(12, 12))

    plt.title("GP Regression")
    plt.fill_between(X_test.flatten(), mean_pred.flatten()-2*std, mean_pred.flatten()+2*std, label='$\pm$2 standard deviations of posterior', color="#dddddd")
    plt.plot(X_underlying, y_underlying, 'b-', label='Underlying function')
    plt.plot(X_test, mean_pred, 'r-', label='Mean of posterior')  
    plt.plot(X_train, y_train, 'kx', ms=8 ,label='Training data')
    plt.legend()
    plt.show()

   


def main():

    X_train, y_train, X_test, _, X_underlying, y_underlying = get_data()

    # Initialize the kernel and Gaussian Process
    kernel = SquaredExponential(l=50.0, sigma_f=1.0)
    kernel = SquaredExponentialPlusPeriodic(l=50.0, sigma_f=1.0, p=100.0)
    kernel = Periodic(l=50.0, sigma_f=1.0, p=100.0)
    kernel = SquaredExponentialTimesPeriodic(l=50.0, sigma_f=1.0, p=100.0)

    gp = GaussianProcess(kernel)

    # Fit the GP to the training data
    gp.fit(X_train, y_train)

    # Optimize hyperparameters
    gp.optimize_hyperparameters(learning_rate=0.5, n_iters=100)

    # Predict the missing values
    mean_pred, cov_pred = gp.predict(X_test)

    # Plot the results using the covariance predictions as fill between
    plot_gp(X_train, y_train, X_test, X_underlying, y_underlying, mean_pred, cov_pred)

        
       

if __name__ == "__main__":
    main()  

