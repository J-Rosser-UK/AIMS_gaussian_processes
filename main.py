import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from load_data import get_data
from covariance_functions import CovarianceFunction, SquaredExponential, Periodic, SquaredExponentialPlusPeriodic
    


class GaussianProcess:
    def __init__(self, kernel: CovarianceFunction, noise_variance: float=0.2, jitter:float = 1e-8):   
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.jitter = jitter

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor):
        self.X_train = X_train
        self.y_train = y_train

        # Compute the covariance matrix of the training points
        K = self.kernel(X_train, X_train)
        
        # Add noise to the diagonal
        noise_variance = self.noise_variance * torch.eye(X_train.size(0))
        K += noise_variance

        # Add jitter to the diagonal
        jitter = self.jitter * torch.eye(X_train.size(0))
        K += jitter

        print(np.linalg.cond(K))
        
        # Cholesky decomposition
        self.K_inv = torch.inverse(K)

    def predict(self, X_test: torch.Tensor):
        # Covariance between test points and training points
        K_star = self.kernel(X_test, self.X_train)
        K_star_star = self.kernel(X_test, X_test)
        
        # Compute the predictive mean
        mean_s = K_star @ self.K_inv @ self.y_train
        
        # Compute the predictive covariance
        cov_s = K_star_star - K_star @ self.K_inv @ K_star.T
        
        return mean_s, cov_s
    

    

def plot_gp(X_train, y_train, X_test, X_underlying, y_underlying, mean_pred, cov_pred):
    std = np.sqrt(np.diag(cov_pred)) 
   
    
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
    kernel = SquaredExponentialPlusPeriodic(l=50.0, sigma_f=1.0)
    kernel = Periodic(p=1000.0, sigma_f=1.0)
    gp = GaussianProcess(kernel)

    # Fit the GP to the training data
    gp.fit(X_train, y_train)

    # Predict the missing values
    mean_pred, cov_pred = gp.predict(X_test)

    # Plot the results using the covariance predictions as fill between
    plot_gp(X_train, y_train, X_test, X_underlying, y_underlying, mean_pred, cov_pred)

        


       

if __name__ == "__main__":
    main()  

