import multiprocessing as mp
import pandas as pd
import numpy as np
import os,sys
import math
from CODE.src.utils import paths

NUM_ZONES =100
NUM_QUARTERS = 22
WORK_ORDER_TYPE = "Ballast Undercutting"
LINE = "C195"
INPUT_FILE = "/Users/rahul.chowdhury/Downloads/TRC_joined_16nov/TRC_"+LINE[1:]+"_16nov/combined_"+LINE[1:]+".csv"
topl_total =pd.DataFrame()
topr_total=pd.DataFrame()
twist3_total =pd.DataFrame()
combined_total =pd.DataFrame()
topl_count =pd.DataFrame()
topr_count=pd.DataFrame()
twist3_count =pd.DataFrame()
combined_count =pd.DataFrame()
metric_to_column_mapping = dict()

def load_data_from_file(filename):

    return pd.read_csv(filename)

def get_zone_length(sorted_metrages):
    return (sorted_metrages[-1]-sorted_metrages[0])/NUM_ZONES

def get_quarter(date):
    year,month,day = date.split("-")
    year = int(year)
    month = int(month)
    year_coeff = year-2014
    month_coeff = 0
    if year==2014:
        if month==2:
            month_coeff = 1
        elif month==6:
            month_coeff = 2
        elif month==10:
            month_coeff = 3
        elif month==12:
            month_coeff = 4
    elif year==2015:
        if month==2:
            month_coeff = 1
        elif month==6:
            month_coeff = 2
        elif month==8:
            month_coeff = 3
        elif month==11 or month==12:
            month_coeff = 4
    elif year==2016:
        if month==2 or month==3:
            month_coeff = 1
        elif month==6:
            month_coeff = 2
        elif month==9:
            month_coeff = 3
        elif month==10:
            month_coeff = 4
    elif year==2017:
        if month==4 :
            month_coeff = 1
        elif month==6:
            month_coeff = 2
        elif month==9:
            month_coeff = 3
        elif month==11:
            month_coeff = 4
    elif year==2018:
        if month==3 :
            month_coeff = 1
        elif month==6:
            month_coeff = 2
        elif month==8:
            month_coeff = 3
        elif month==10:
            month_coeff = 4
    elif year==2019:
        if month==3 :
            month_coeff = 1
        elif month==5:
            month_coeff = 2
    return year_coeff*4+month_coeff


def just_wait_and_print_len_and_idx(data):
    """Waits for 5 seconds and prints df length and first and last index"""
    # Extract some info
    count = 1
    filenum = 0
    for i, row in data.iterrows():

        if count == 1:
            filenum = str(int(i / 100000))
        count += 1

        try:

            zone = "zone" + str(math.ceil((row['METRAGE'] - first_metrage) / zone_length))
            if int(zone[4:])<1 or int(zone[4:])>100:
                continue
            quarter = "quarter" + str(get_quarter(row["Date"]))
            topl_value = row['TOP L']
            topr_value = row['TOP R']
            twist3_value = row['TW 3']
            combined_value = float('nan') if (
                    np.isnan(topl_value) or np.isnan(topr_value) or np.isnan(twist3_value)) else (
                    (topl_value + topr_value) / 2.0 + twist3_value)
            topl_total.loc[zone][quarter] += topl_value**3
            topr_total.loc[zone][quarter] += topr_value**3
            twist3_total.loc[zone][quarter] += twist3_value**3
            combined_total.loc[zone][quarter] += combined_value**3

            topl_count.loc[zone][quarter] += 1
            topr_count.loc[zone][quarter] += 1
            twist3_count.loc[zone][quarter] += 1
            combined_count.loc[zone][quarter] += 1
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    if not os.path.exists(paths.TRC_ML + "/trc/" + str(LINE) + "/combined_zonal_cubic"):
        os.makedirs(paths.TRC_ML + "/trc/" + str(LINE) + "/combined_zonal_cubic")

    if not os.path.exists(paths.TRC_ML + "/trc/" + str(LINE) + "/topl_zonal_cubic"):
        os.makedirs(paths.TRC_ML + "/trc/" + str(LINE) + "/topl_zonal_cubic")

    if not os.path.exists(paths.TRC_ML + "/trc/" + str(LINE) + "/topr_zonal_cubic"):
        os.makedirs(paths.TRC_ML + "/trc/" + str(LINE) + "/topr_zonal_cubic")
    if not os.path.exists(paths.TRC_ML + "/trc/" + str(LINE) + "/twist3_zonal_cubic"):
        os.makedirs(paths.TRC_ML + "/trc/" + str(LINE) + "/twist3_zonal_cubic")


    combined_total.to_pickle(paths.TRC_ML + "/trc/" + str(LINE) + "/combined_zonal_cubic/part_" + filenum + ".pkl")
    topl_total.to_pickle(paths.TRC_ML + "/trc/" + str(LINE) + "/topl_zonal_cubic/part_" + filenum + ".pkl")
    topr_total.to_pickle(paths.TRC_ML + "/trc/" + str(LINE) + "/topr_zonal_cubic/part_" + filenum + ".pkl")
    twist3_total.to_pickle(paths.TRC_ML + "/trc/" + str(LINE) + "/twist3_zonal_cubic/part_" + filenum + ".pkl")

    # combined_count.to_pickle(paths.TRC_ML + "/trc/" + str(LINE) + "/combined_zonal_linear/part_" + filenum + ".pkl")
    # topl_count.to_pickle(paths.TRC_ML + "/trc/" + str(LINE) + "/topl_zonal_linear/part_" + filenum + ".pkl")
    # topr_count.to_pickle(paths.TRC_ML + "/trc/" + str(LINE) + "/topr_zonal_linear/part_" + filenum + ".pkl")
    # twist3_count.to_pickle(paths.TRC_ML + "/trc/" + str(LINE) + "/twist3_zonal_linear/part_" + filenum + ".pkl")




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

def init(combined_total_,topl_total_,topr_total_,twist3_total_,combined_count_,topl_count_,topr_count_,twist3_count_,
                                                      first_metrage_,
                                                      zone_length_):
    global combined_total
    global twist3_total
    global topl_total
    global topr_total
    combined_total = combined_total_
    twist3_total = twist3_total_
    topl_total = topl_total_
    topr_total = topr_total_

    global combined_count
    global twist3_count
    global topl_count
    global topr_count
    combined_count = combined_count_
    twist3_count= twist3_count_
    topl_count = topl_count_
    topr_count = topr_count_

    global first_metrage
    first_metrage = first_metrage_  # must be inherited, not passed as an argument

    global zone_length
    zone_length = zone_length_  # must be inherited, not passed as an argument

def main():
    data = load_data_from_file(INPUT_FILE)
    dates = data["Date"].unique()
    sorted_dates = sorted(dates, key=lambda d: tuple(map(int, d.split('-'))))
    metrages = data["METRAGE"].unique()
    sorted_metrages = sorted(metrages)
    zone_length = get_zone_length(sorted_metrages)
    quarters = ["quarter" + str(i) for i in range(1, NUM_QUARTERS+1)]
    zones = ["zone" + str(i) for i in range(1, NUM_ZONES+1)]
    first_metrage=sorted_metrages[0]-0.000001

    topl_total = pd.DataFrame(0, columns=quarters, index=zones)
    topr_total = pd.DataFrame(0, columns=quarters, index=zones)
    twist3_total = pd.DataFrame(0, columns=quarters, index=zones)
    combined_total = pd.DataFrame(0, columns=quarters, index=zones)

    topl_count = pd.DataFrame(0, columns=quarters, index=zones)
    topr_count = pd.DataFrame(0, columns=quarters, index=zones)
    twist3_count = pd.DataFrame(0, columns=quarters, index=zones)
    combined_count = pd.DataFrame(0, columns=quarters, index=zones)

    # Job parameters
    n_jobs = mp.cpu_count()  # Poolsize
    size = (100, 1000)  # Size of DataFrame
    chunksize = 100000  # Maximum size of Frame Chunk

    # shared_arr = mp.Array(ctypes.c_double, 10)
    # first_metrage = mp.Value(ctypes.c_float,sorted_metrages[0])
    # zone_length = mp.Value(ctypes.c_float,get_zone_length(sorted_metrages))

    # Preparation
    df = pd.DataFrame(np.random.rand(*size))
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(n_jobs,initializer=init,initargs=(combined_total,topl_total,topr_total,twist3_total,combined_count,topl_count,topr_count,twist3_count,
                                                      first_metrage,
                                                      zone_length,))

    print('Starting MP')

    # Execute the wait and print function in parallel
    pool.map_async(just_wait_and_print_len_and_idx, df_chunking(data, chunksize))

    pool.close()
    pool.join()
    print('DONE')
    # # print(a)

if __name__ == '__main__':
    main()