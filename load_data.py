import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from datetime import timedelta

def pad_dataframe(df, pad_rows=100, time_delta=timedelta(minutes=5)):
    # Get the time range to pad before and after
    start_time = df['Reading Date and Time (ISO)'].min() - time_delta * pad_rows
    end_time = df['Reading Date and Time (ISO)'].max() + time_delta * pad_rows

    # Generate the complete time index
    complete_time_range = pd.date_range(start=start_time, end=end_time, freq=time_delta)

    # Create a new DataFrame with NaN values and the complete time range
    padded_df = pd.DataFrame({
        'Reading Date and Time (ISO)': complete_time_range,
        'Tide height (m)': [np.nan] * len(complete_time_range),
        'True tide height (m)': [np.nan] * len(complete_time_range)
    })

    # Merge the original data into the padded DataFrame
    padded_df = padded_df.merge(df, on='Reading Date and Time (ISO)', how='left', suffixes=('', '_original'))

    # Replace NaN in original columns with the values from the merged columns
    padded_df['Tide height (m)'] = padded_df['Tide height (m)_original'].combine_first(padded_df['Tide height (m)'])
    padded_df['True tide height (m)'] = padded_df['True tide height (m)_original'].combine_first(padded_df['True tide height (m)'])

    # Drop the extra columns
    padded_df = padded_df[['Reading Date and Time (ISO)', 'Tide height (m)', 'True tide height (m)']]
    
    return padded_df

def get_data():

    # Load the data into a pandas DataFrame
    df = pd.read_csv('sotonmet.txt', parse_dates=['Update Date and Time (ISO)', 'Reading Date and Time (ISO)'])

    # Drop duplicates in reading date and time
    df.drop_duplicates(inplace=True, subset='Reading Date and Time (ISO)')

    # Remove all columns except index, tide height and true tide height
    df = df[['Reading Date and Time (ISO)', 'Tide height (m)', 'True tide height (m)']]

    df = pad_dataframe(df)  

    # Convert datetime to be minutes since the first reading as an integer
    df['Reading Date and Time (ISO)'] = (df['Reading Date and Time (ISO)'] - df['Reading Date and Time (ISO)'][0]).dt.total_seconds() / 60

    df.set_index('Reading Date and Time (ISO)', inplace=True)

    print(df.head())   

    tide_height = df['Tide height (m)'].values
    true_tide_height = df['True tide height (m)'].values
    reading_times = df.index.values
    
   
    # Turn this into an array of floats eg [[reading_times_1, tide_height_1], [reading_times_2, tide_height_2], ...]
    tide_height_data = torch.tensor(list(zip(reading_times, tide_height)), dtype=torch.float32)

    # Let the train data be points where the tide height is not NaN
    X_train, y_train = tide_height_data[~torch.isnan(tide_height_data[:, 1])].split(1, dim=1)

    print(X_train)
    

    # Let the test data be points where the tide height is NaN
    X_test, _ = tide_height_data[torch.isnan(tide_height_data[:, 1])].split(1, dim=1)  


    # Get the true tide height data
    true_tide_height = torch.tensor(list(zip(reading_times, df['True tide height (m)'].values)), dtype=torch.float32)
    
    print(true_tide_height)
    # Remove any rows with NaN values
    true_tide_height = true_tide_height[~torch.isnan(true_tide_height[:, 1])]
    print(true_tide_height)

    # Get the underlying data
    X_underlying, y_underlying = true_tide_height.split(1, dim=1)


    # centre the y axis data
    y_underlying = y_underlying - y_underlying.mean()

    y_train = y_train - y_train.mean()

    

    return X_train, y_train, X_test, _, X_underlying, y_underlying


def plot_samples_and_underlying_data(X_train, y_train, X_underlying, y_underlying):
    """
        Plot the training data as crosses and the underlying distribution as a solid line
        on the same plot.
    """
    figure = plt.figure(figsize=(10, 6))
    plt.plot(X_train, y_train, 'kx', ms=8 ,label='Training data')
    plt.plot(X_underlying, y_underlying, 'b-', label='Underlying data')
    plt.legend()

    plt.show()


def main():
    X_train, y_train, X_test, _, X_underlying, y_underlying = get_data()
    # print(X_train)
    plot_samples_and_underlying_data(X_train, y_train, X_underlying, y_underlying)


if __name__ == '__main__':
    main()