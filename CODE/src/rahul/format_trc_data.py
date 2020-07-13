import pandas as pd
import numpy as np
from CODE.src.utils import paths

input_file = "/Users/rahul.chowdhury/Downloads/TRC_joined_16nov/TRC_195_16nov/combined_195.csv"

data = pd.read_csv(input_file)
# print(data.head())

dates = data["Date"].unique()
sorted_dates = sorted(dates, key=lambda d:tuple(map(int, d.split('-'))))


metrages = data["METRAGE"].unique()
sorted_metrages = sorted(metrages)

topL = pd.DataFrame(columns=sorted_dates,index=sorted_metrages)
topR = pd.DataFrame(columns=sorted_dates,index=sorted_metrages)
twist3 = pd.DataFrame(columns=sorted_dates,index=sorted_metrages)
combined = pd.DataFrame(columns=sorted_dates,index=sorted_metrages)

for i, row in data.iterrows():
    try:
        topl_value =  row['TOP L']
        topr_value =  row['TOP R']
        twist3_value =  row['TW 3']
        combined_value = float('nan')  if (np.isnan(topl_value) or np.isnan(topr_value) or np.isnan(twist3_value)) else (topl_value + topr_value/2.0 + twist3_value)
        topL.loc[row['METRAGE']][row['Date']] = topl_value
        topR.loc[row['METRAGE']][row['Date']] = topr_value
        twist3.loc[row['METRAGE']][row['Date']] = twist3_value
        combined.loc[row['METRAGE']][row['Date']] = combined_value
    except Exception as e:
        print(e)

topL.to_pickle(paths.TRC_ML + "/C195 topL with null.pkl")
topL_null_removed = topL.ffill(axis=1).bfill(axis=1)
topL_null_removed.to_pickle(paths.TRC_ML + "/C195 topL without null.pkl")

topR.to_pickle(paths.TRC_ML + "/C195 topR with null.pkl")
topR_null_removed = topR.ffill(axis=1).bfill(axis=1)
topR_null_removed.to_pickle(paths.TRC_ML + "/C195 topR without null.pkl")

twist3.to_pickle(paths.TRC_ML + "/C195 twist3 with null.pkl")
twist3_null_removed = twist3.ffill(axis=1).bfill(axis=1)
twist3_null_removed.to_pickle(paths.TRC_ML + "/C195 twist3 without null.pkl")

combined.to_pickle(paths.TRC_ML + "/C195 combined with null.pkl")
combined_null_removed = combined.ffill(axis=1).bfill(axis=1)
combined_null_removed.to_pickle(paths.TRC_ML + "/C195 combined without null.pkl")
