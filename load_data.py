import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a pandas DataFrame
df = pd.read_csv('sotonmet.txt', parse_dates=['Update Date and Time (ISO)', 'Reading Date and Time (ISO)'])

# Set the 'Reading Date and Time (ISO)' as the index
df.set_index('Reading Date and Time (ISO)', inplace=True)

# Create subplots
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
fig.suptitle('Sotonmet Data Over Time')

# Print a list of column names
for col in df.columns:
    print(col)

# List of columns to plot (excluding those that contain '(ISO)' in the name)
columns_to_plot = [col for col in df.columns if '(ISO)' not in col]

# List all columns with both tide and height in the name
columns_to_plot = [col for col in df.columns if 'tide' in col.lower() and 'height' in col.lower()]

# Plot each column
for ax, column in zip(axs.flatten(), columns_to_plot):
    ax.plot(df.index, df[column])
    ax.set_title(column)
    ax.set_xlabel('Time')
    ax.set_ylabel(column)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

