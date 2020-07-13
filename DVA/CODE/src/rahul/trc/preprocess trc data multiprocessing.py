import multiprocessing as mp
import pandas as pd
import numpy as np
import os
import sys
import shutil
from CODE.src.utils import paths

LINE = "C195"
INPUT_FILE = "/Users/rahul.chowdhury/Downloads/TRC_joined_16nov/TRC_"+LINE[1:]+"_16nov/combined_"+LINE[1:]+".csv"
a = [0]*10
combined =pd.DataFrame()



def load_data_from_file(filename):
    return pd.read_csv(filename)

def delete_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def process_partial_data(data,):
    """Waits for 5 seconds and prints df length and first and last index"""
    # Extract some info
    count = 1
    filenum = ""
    for i, row in data.iterrows():

        if count==1:
            filenum = str(int(i/100000))
        count +=1

        try:
            topl_value = row['TOP L']
            topr_value = row['TOP R']
            twist3_value = row['TW 3']
            combined_value = float('nan') if (
                        np.isnan(topl_value) or np.isnan(topr_value) or np.isnan(twist3_value)) else (
                (topl_value + topr_value )/ 2.0 + twist3_value)

            combined.loc[row['METRAGE']][row['Date']] += combined_value
            # combined.loc[row['METRAGE']][row['Date']] += topl_value
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    combined.to_pickle(paths.TRC_ML + "/trc/" + str(LINE) + "/combined/part_" + filenum + ".pkl")





def df_chunking(df, chunksize):
    """Splits df into chunks, drops data of original df inplace"""
    count = 0  # Counter for chunks
    while len(df):
        count += 1
        print('Preparing chunk {}'.format(count))
        # Return df chunk
        yield df.iloc[:chunksize].copy()
        # Delete data in place because it is no longer needed
        df.drop(df.index[:chunksize], inplace=True)

def init(combined_):
    global combined
    combined = combined_


def main():
    data = load_data_from_file(INPUT_FILE)
    dates = data["Date"].unique()
    sorted_dates = sorted(dates, key=lambda d: tuple(map(int, d.split('-'))))
    metrages = data["METRAGE"].unique()
    sorted_metrages = sorted(metrages)

    combined = pd.DataFrame(columns=sorted_dates, index=sorted_metrages)

    # Job parameters
    n_jobs = mp.cpu_count()
    chunksize = 100000  # Maximum size of Frame Chunk

    # shared_arr = mp.Array(ctypes.c_double, 10)
    # first_metrage = mp.Value(ctypes.c_float,sorted_metrages[0])
    # zone_length = mp.Value(ctypes.c_float,get_zone_length(sorted_metrages))

    # Preparation
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(n_jobs,initializer=init,initargs=(combined,))

    print('Starting MP')

    # Execute the wait and print function in parallel
    pool.map_async(process_partial_data, df_chunking(data, chunksize))

    pool.close()
    pool.join()
    print('DONE')
    # # print(a)

if __name__ == '__main__':
    main()