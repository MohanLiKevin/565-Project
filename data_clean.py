import pandas as pd

# Load the dataset
df = pd.read_csv('Popularity.csv')

# Filter out rows where the tag is '[None]'
df_filtered = df[df['tags'] != '[None]']

# Save the filtered DataFrame to a new CSV file
filtered_file_path = 'Popularity_filtered.csv'
df_filtered.to_csv(filtered_file_path, index=False)

filtered_file_path
