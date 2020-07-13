import pandas as pd

df = pd.read_excel("/Users/rahul.chowdhury/omscs/DVA/project/dva_project/data/ZR0357-15-XLS01-C QR 2015 RASC Survey (Brisbane) - Trackbed Metrics.xlsx", skiprows=1000)
print(df.head())