import multiprocessing as mp
import pandas as pd
import os
import sys
import shutil
from CODE.src.utils import paths

LINE = "C138"
INPUT_FILE = "/Users/rahul.chowdhury/Downloads/C"+LINE[1:]+"_GPR_TRC_nov20/combined_"+LINE[1:]+".csv"
a = [0]*10
pvc_left =pd.DataFrame()
pvc_centre=pd.DataFrame()
pvc_right =pd.DataFrame()
pvc_combined =pd.DataFrame()
metric_to_column_mapping = dict()


def get_metric_to_column_mapping():
    df = pd.read_excel(paths.DATA + "/Mapping GPR Column Names to GPR Features.xlsx")
    metric_to_column = dict()
    for index, row in df.iterrows():
        metric_to_column[row["GPR Feature"]] = row["GPR Column Name"]

    return metric_to_column


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

            if count == 1:
        filenum = str(int(i / 100000))
    count += 1

    try:
        if row[metric_to_column_mapping['PVCLeft']] == "<ufunc 'isnan'>" or \
                row[metric_to_column_mapping['PVCCentre']] == "<ufunc 'isnan'>" or \
                row[metric_to_column_mapping['PVCRight']] == "<ufunc 'isnan'>":
            continue
        pvc_left_value = float(row[metric_to_column_mapping['PVCLeft']])
        pvc_centre_value = float(row[metric_to_column_mapping['PVCCentre']])
        pvc_right_value = float(row[metric_to_column_mapping['PVCRight']])
        combined_value = pvc_left_value + pvc_centre_value + pvc_right_value
        pvc_left.loc[row['METRAGE']][row['Date']] = pvc_left_value
        pvc_centre.loc[row['METRAGE']][row['Date']] = pvc_centre_value
        pvc_right.loc[row['METRAGE']][row['Date']] = pvc_right_value
        pvc_combined.loc[row['METRAGE']][row['Date']] = combined_value
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    pvc_left.to_pickle(paths.TRC_ML + "/gpr/" + str(LINE) + "/pvc_left/part_" + filenum + ".pkl")
    pvc_right.to_pickle(paths.TRC_ML + "/gpr/" + str(LINE) + "/pvc_right/part_" + filenum + ".pkl")
    pvc_centre.to_pickle(paths.TRC_ML + "/gpr/" + str(LINE) + "/pvc_centre/part_" + filenum + ".pkl")
    pvc_combined.to_pickle(paths.TRC_ML + "/gpr/" + str(LINE) + "/pvc_combined/part_" + filenum + ".pkl")





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

def init(pvc_left_,
         pvc_right_,
         pvc_centre_,
         pvc_combined_,
         metric_to_column_mapping_):
    global pvc_left
    global pvc_right
    global pvc_combined
    global pvc_centre
    global metric_to_column_mapping
    pvc_left = pvc_left_
    pvc_right = pvc_right_
    pvc_combined = pvc_combined_
    pvc_centre = pvc_centre_
    metric_to_column_mapping = metric_to_column_mapping_


def main():
    data = load_data_from_file(INPUT_FILE)
    dates = data["Date"].unique()
    sorted_dates = sorted(dates, key=lambda d: tuple(map(int, d.split('-'))))
    metrages = data["METRAGE"].unique()
    sorted_metrages = sorted(metrages)
    metric_to_column = get_metric_to_column_mapping()

    pvc_left = pd.DataFrame(columns=sorted_dates, index=sorted_metrages)
    pvc_right = pd.DataFrame(columns=sorted_dates, index=sorted_metrages)
    pvc_centre = pd.DataFrame(columns=sorted_dates, index=sorted_metrages)
    pvc_combined = pd.DataFrame(columns=sorted_dates, index=sorted_metrages)

    # Job parameters
    n_jobs = mp.cpu_count()
    chunksize = 100000  # Maximum size of Frame Chunk

    # shared_arr = mp.Array(ctypes.c_double, 10)
    # first_metrage = mp.Value(ctypes.c_float,sorted_metrages[0])
    # zone_length = mp.Value(ctypes.c_float,get_zone_length(sorted_metrages))

    # Preparation
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(n_jobs,initializer=init,initargs=(pvc_left,
                                                      pvc_right,
                                                      pvc_centre,
                                                      pvc_combined,
                                                      metric_to_column,))

    print('Starting MP')

    # Execute the wait and print function in parallel
    pool.map_async(process_partial_data, df_chunking(data, chunksize))

    pool.close()
    pool.join()
    print('DONE')
    # # print(a)

if __name__ == '__main__':
    main()