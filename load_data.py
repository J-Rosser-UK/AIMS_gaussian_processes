import pandas as pd
import matplotlib.pyplot as plt
import torch

def get_data():

    # Load the data into a pandas DataFrame
    df = pd.read_csv('sotonmet.txt', parse_dates=['Update Date and Time (ISO)', 'Reading Date and Time (ISO)'])


    df.drop_duplicates(inplace=True, subset='Reading Date and Time (ISO)')

    # Set the 'Reading Date and Time (ISO)' as the index
    df.set_index('Reading Date and Time (ISO)', inplace=True)

    # Drop duplicates in reading date and time

    # Convert reading times into a datetime object
    reading_times = pd.to_datetime(df.index.values)

    # Convert reading times to minutes since the first reading
    reading_times = (reading_times - reading_times[0]).total_seconds()/60


    # update index in dataframe
    df.index = reading_times

    tide_height, reading_times = df['Tide height (m)'].values, df.index.values
   
    # Turn this into an array of floats eg [[reading_times_1, tide_height_1], [reading_times_2, tide_height_2], ...]
    tide_height_data = torch.tensor(list(zip(reading_times, tide_height)), dtype=torch.float32)

    # Let the train data be points where the tide height is not NaN
    X_train, y_train = tide_height_data[~torch.isnan(tide_height_data[:, 1])].split(1, dim=1)
    

    # Let the test data be points where the tide height is NaN
    X_test, _ = tide_height_data[torch.isnan(tide_height_data[:, 1])].split(1, dim=1)  

    # Get the underlying data
    y_underlying, X_underlying = df['True tide height (m)'].values, df.index.values

    # centre the y axis data
    y_underlying = y_underlying - y_underlying.mean()

    y_train = y_train - y_train.mean()


    return X_train, y_train, X_test, _, X_underlying, y_underlying


def plot_samples_and_underlying_data(X_train, y_train, X_underlying, y_underlying):
    """
        Plot the training data as crosses and the underlying distribution as a solid line
        on the same plot.
    """
    plt.plot(X_train, y_train, 'kx', ms=8 ,label='Training input-target pairs $\{(x_i, f_i | i = 1,...,n)\}$')
    plt.plot(X_underlying, y_underlying, 'b-', label='Underlying distribution')
    plt.legend()

    plt.show()


def main():
    X_train, y_train, X_test, _, X_underlying, y_underlying = get_data()
    print(X_train)
    plot_samples_and_underlying_data(X_train, y_train, X_underlying, y_underlying)


if __name__ == '__main__':
    main()