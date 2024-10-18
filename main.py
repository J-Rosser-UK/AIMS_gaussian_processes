import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from load_data import get_data
from kernels import CovarianceFunction, SquaredExponential, Periodic, SquaredExponentialPlusPeriodic, SquaredExponentialTimesPeriodic
from tqdm import tqdm
from pprint import pprint
   

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

        print("Condition Number: ", np.linalg.cond(K.detach().numpy()))
        
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

        optimizer = torch.optim.Adam(self.kernel.parameters(), lr=learning_rate)
        
        for i in tqdm(range(n_iters)):
            optimizer.zero_grad()  
            loss = -self.log_marginal_likelihood()  
            loss.backward()  
            optimizer.step() 
            
            if i % 10 == 0:
                print(f"Iteration {i+1}/{n_iters} - Loss: {loss.item()}")

        print(f"Final loss: {loss.item()}")
        final_params = {name: param.item() for name, param in self.kernel.named_parameters()}
        pprint(f"Final parameters: {final_params}")

    
    def plot_draws_from_gp(self, X_test, n_draws=10):

        # Predict the mean and covariance on test data
        mean_s, cov_s = self.predict(X_test)

        mean_s = mean_s.detach().numpy()
        cov_s = cov_s.detach().numpy()

        # Generate samples from the predictive distribution
        n_samples = 5  # Number of random draws to plot
        samples = np.random.multivariate_normal(mean_s.flatten(), cov_s, n_samples)

        # Plot the results
        plt.figure(figsize=(16, 6))
        plt.plot(X_test.numpy(), mean_s, 'b-', lw=2, label='Predictive mean')

        for i in range(n_samples):
            plt.plot(X_test.numpy(), samples[i], lw=1.5, label=f'Sample {i+1}')

        plt.legend()
        plt.title('Draws from the GP posterior')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()
        

    def plot_gp(self, X_train, y_train, X_test, X_underlying, y_underlying, mean_pred, cov_pred):
        std = np.sqrt(np.diag(cov_pred.detach().numpy())) 
        mean_pred = mean_pred.detach().numpy()

        fig = plt.figure(figsize=(16,6))

        plt.title("Gaussian Process Regression")
        
        # Plot ±2 standard deviations with lighter shade
        plt.fill_between(X_test.flatten(), 
                        mean_pred.flatten() - 2 * std, 
                        mean_pred.flatten() + 2 * std, 
                        label='$\pm$2 standard deviations of posterior', 
                        color="#dddddd")
        
        # Plot ±1 standard deviation with a slightly darker shade
        plt.fill_between(X_test.flatten(), 
                        mean_pred.flatten() - std, 
                        mean_pred.flatten() + std, 
                        label='$\pm$1 standard deviation of posterior', 
                        color="#bbbbbb")
        
        # Plot underlying function, posterior mean, and training data
        plt.plot(X_underlying, y_underlying, 'b-', label='Underlying function')
        plt.plot(X_test, mean_pred, 'r-', label='Mean of posterior')  
        plt.plot(X_train, y_train, 'kx', ms=8, label='Training data')
        
        plt.legend()
        plt.show()

   


def main():

    X_train, y_train, X_test, _, X_underlying, y_underlying = get_data()

    # Initialize the kernel and Gaussian Process
    kernel = SquaredExponential(l=50.0, sigma_f=1.0)
    # # kernel = SquaredExponentialPlusPeriodic(l=50.0, sigma_f=1.0, p=100.0, omega=1.0)
    # kernel = Periodic(omega=1.0, sigma_f=1.0, p=500.0)
    # kernel = SquaredExponentialTimesPeriodic(l=50.0, sigma_f=1.0, p=500.0, omega=1.0)

    gp = GaussianProcess(kernel)

    # Fit the GP to the training data
    gp.fit(X_train, y_train)

    # Optimize hyperparameters
    # gp.optimize_hyperparameters(learning_rate=0.5, n_iters=100)

    # Predict the missing values
    mean_pred, cov_pred = gp.predict(X_test)

    gp.plot_draws_from_gp(X_test, n_draws=10)

    # Plot the results using the covariance predictions as fill between
    gp.plot_gp(X_train, y_train, X_test, X_underlying, y_underlying, mean_pred, cov_pred)

        
       

if __name__ == "__main__":
    main()  

