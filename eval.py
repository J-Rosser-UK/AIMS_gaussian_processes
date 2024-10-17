from load_data import get_data
import torch
from kernels import SquaredExponential
from main import GaussianProcess

def root_mean_squared_error(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred)**2))


def main():

    X_train, y_train, X_test, _, X_underlying, y_underlying = get_data()

    # Initialize the kernel and Gaussian Process
    kernel = SquaredExponential(l=50.0, sigma_f=1.0)
    gp = GaussianProcess(kernel)

    # Fit the GP to the training data
    gp.fit(X_train, y_train)

    # Predict the missing values
    mean_pred, cov_pred = gp.predict(X_test)

    # Calculate the root mean squared error between the underlying data and the mean_pred by first aligning the two
    # datasets so that their x values are the same
    
    underlying_dataset = torch.tensor(list(zip(X_underlying, y_underlying)), dtype=torch.float32)
    predictions_dataset = torch.tensor(list(zip(X_test, mean_pred)), dtype=torch.float32)

    print(underlying_dataset.shape)
    print(predictions_dataset.shape)

    print(underlying_dataset)

    # Find the indices of the underlying dataset that are in the predictions dataset
    indices = torch.tensor([i for i in range(underlying_dataset.shape[0]) if underlying_dataset[i, 0] in predictions_dataset[:, 0]])
    print(indices)

    # Remove any indices that are not in the predictions dataset
    underlying_dataset = underlying_dataset[indices]

    print(underlying_dataset.shape)
    print(predictions_dataset.shape)

    # Calculate the root mean squared error
    rmse = root_mean_squared_error(underlying_dataset[:, 1], predictions_dataset[:, 1])
    print(f"Root mean squared error: {rmse.item()}")


    
    



if __name__ == "__main__":

    main()