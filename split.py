import pandas as pd
import random

# Define the file paths
input_file = './data/eit/24x24_Images_11Cond_30k_2022-02-23.csv'
train_file = './data/eit/train_images.csv'
test_file = './data/eit/test_images.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(input_file)

# Set a seed for reproducibility
seed = 42
random.seed(seed)

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=seed)

# Calculate the split indices based on the 80/20 ratio
split_ratio = 0.8
split_index = int(len(df) * split_ratio)

# Split the DataFrame into train and test sets
train_data = df[:split_index]
test_data = df[split_index:]

# Save the train and test sets to separate CSV files
train_data.to_csv(train_file, index=False)
test_data.to_csv(test_file, index=False)

print("Data split and saved successfully.")

