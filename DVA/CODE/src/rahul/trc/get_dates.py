import pandas as pd

input_file = "/Users/rahul.chowdhury/Downloads/TRC_joined_16nov/TRC_195_16nov/combined_195.csv"

data = pd.read_csv(input_file)
# print(data.head())

dates = data["Date"].unique()
sorted_dates = set(sorted(dates, key=lambda d:tuple(map(int, d.split('-')))))


input_file = "/Users/rahul.chowdhury/Downloads/TRC_joined_16nov/TRC_138_16nov/combined_138.csv"

data = pd.read_csv(input_file)
# print(data.head())

dates = data["Date"].unique()
sorted_dates.update(set(sorted(dates, key=lambda d:tuple(map(int, d.split('-'))))))
print('\n'.join(sorted(sorted_dates)))
