import numpy as np
import pandas as pd

# load the data
df = pd.read_csv("data/placement.csv")

# first look at the data
print("Shape (rows, columns):", df.shape)
print()
print("First 5 rows:")
print(df.head())
print()
print("Column names:", list(df.columns))
print()
print("Data types and null values:")
print(df.info())
print()
print("Summary statistics:")
print(df.describe())

# numpy on the first numeric column
num_cols = df.select_dtypes(include="number").columns
first_col = num_cols[0]
values = df[first_col].to_numpy()

print()
print("Numpy check on column:", first_col)
print("Mean:", np.mean(values))
print("Median:", np.median(values))
print("Std dev:", np.std(values))