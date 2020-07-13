from multiprocessing import Pool, Value
import pandas as pd
import os
import numpy as np
from CODE.src.utils import paths

num_partitions = 10 #number of partitions to split dataframe
num_cores = os.cpu_count() #number of cores on your machine

# modified_df = pd.DataFrame()
# counter = None



def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def process_image(data):
    for i, row in data.iterrows():
        try:
            modified_df.loc[row['METRAGE']][row['Date']] = row['TOP L']

        except Exception as e:
            print(e)

if __name__ == '__main__':



    input_file = "/Users/rahul.chowdhury/Downloads/TRC_joined_16nov/TRC_138_16nov/combined_138.csv"

    data = pd.read_csv(input_file).head(100)
    # print(data.head())

    dates = data["Date"].unique()
    sorted_dates = sorted(dates, key=lambda d: tuple(map(int, d.split('-'))))

    metrages = data["METRAGE"].unique()
    sorted_metrages = sorted(metrages)

    modified_df = pd.DataFrame(columns=sorted_dates, index=sorted_metrages).fillna(0.0)

    num_columns = len(sorted_dates)
    index = 0

    counter = Value('i', 0)

    parallelize_dataframe(data,process_image)
    # for i, row in data.iterrows():
    #     try:
    #         modified_df.loc[row['METRAGE']][row['Date']] = row['TOP L']
    #     except Exception as e:
    #         print(e)
    modified_df.to_pickle(paths.TRC_ML + "/C138.csv")

    # pool = Pool(os.cpu_count())                         # Create a multiprocessing Pool
    # pool.map(process_image, data)  # process data_inputs iterable with pool