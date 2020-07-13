import multiprocessing as mp
import pandas as pd
import numpy as np
import math
from CODE.src.utils import paths

NUM_ZONES =100
NUM_QUARTERS = 22
WORK_ORDER_TYPE = "Ballast Undercutting"
LINE = "C195"
INPUT_FILE = "/Users/rahul.chowdhury/Downloads/TRC_joined_16nov/TRC_195_16nov/combined_195.csv"
a = [0]*10
zone_df_ballast_undercutting =pd.DataFrame()
zone_df_formation_repairs=pd.DataFrame()
zone_df_top_and_line_spot_resurfacing =pd.DataFrame()
zone_df_maintenance_ballasting =pd.DataFrame()
zone_df_mechanised_resleepering=pd.DataFrame()
zone_df_mechanised_resurfacing =pd.DataFrame()


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

        if "," in row["Work_order_type"]:
            pass

        if count==1:
            filenum = str(int(i/100000))
        count +=1
        if "Ballast Undercutting" in row["Work_order_type"]:
            zone = "zone" + str(math.ceil((row['METRAGE'] - first_metrage) / zone_length))
            quarter = "quarter" + str(get_quarter(row["Date"]))
            zone_df_ballast_undercutting.loc[zone][quarter] += 1

        elif "Formation repairs" in row["Work_order_type"]:
            zone = "zone" + str(math.ceil((row['METRAGE'] - first_metrage) / zone_length))
            quarter = "quarter" + str(get_quarter(row["Date"]))
            zone_df_formation_repairs.loc[zone][quarter] += 1

        elif "Top & Line Spot Resurfacing" in row["Work_order_type"]:
            zone = "zone" + str(math.ceil((row['METRAGE'] - first_metrage) / zone_length))
            quarter = "quarter" + str(get_quarter(row["Date"]))
            zone_df_top_and_line_spot_resurfacing.loc[zone][quarter] += 1

        elif "Mechanised Resurfacing" in row["Work_order_type"]:
            zone = "zone" + str(math.ceil((row['METRAGE'] - first_metrage) / zone_length))
            quarter = "quarter" + str(get_quarter(row["Date"]))
            zone_df_mechanised_resurfacing.loc[zone][quarter] += 1

        elif "Maintenance Ballasting" in row["Work_order_type"]:
            zone = "zone" + str(math.ceil((row['METRAGE'] - first_metrage) / zone_length))
            quarter = "quarter" + str(get_quarter(row["Date"]))
            zone_df_maintenance_ballasting.loc[zone][quarter] += 1

        elif "Mechanised Resleepering" in row["Work_order_type"]:
            zone = "zone" + str(math.ceil((row['METRAGE'] - first_metrage) / zone_length))
            quarter = "quarter" + str(get_quarter(row["Date"]))
            zone_df_mechanised_resleepering.loc[zone][quarter] += 1



    zone_df_ballast_undercutting.to_pickle(paths.TRC_ML + "/work_order/" + str(LINE) + "/Ballast Undercutting/" + str(NUM_ZONES) + "/part_" + filenum + ".pkl")
    zone_df_formation_repairs.to_pickle(paths.TRC_ML + "/work_order/" + str(LINE) + "/Formation Repairs/" + str(NUM_ZONES) + "/part_" + filenum + ".pkl")
    zone_df_top_and_line_spot_resurfacing.to_pickle(
        paths.TRC_ML + "/work_order/" + str(LINE) + "/Top & line Spot Resurfacing/" + str(NUM_ZONES) + "/part_" + filenum + ".pkl")
    zone_df_mechanised_resurfacing.to_pickle(
        paths.TRC_ML + "/work_order/" + str(LINE) + "/Mechanised Resurfacing/" + str(NUM_ZONES) + "/part_" + filenum + ".pkl")
    zone_df_maintenance_ballasting.to_pickle(
        paths.TRC_ML + "/work_order/" + str(LINE) + "/Maintenance Ballasting/" + str(NUM_ZONES) + "/part_" + filenum + ".pkl")
    zone_df_mechanised_resleepering.to_pickle(
        paths.TRC_ML + "/work_order/" + str(LINE) + "/Mechanised Resleepering/" + str(NUM_ZONES) + "/part_" + filenum + ".pkl")




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

def init(zone_df_ballast_undercutting_,
                                                      zone_df_formation_repairs_,    zone_df_top_and_line_spot_resurfacing_,
                                                        zone_df_mechanised_resurfacing_,
                                                      zone_df_maintenance_ballasting_,
                                                      zone_df_mechanised_resleepering_,
                                                      first_metrage_,
                                                      zone_length_):
    global zone_df_ballast_undercutting
    global zone_df_formation_repairs
    global zone_df_top_and_line_spot_resurfacing
    global zone_df_maintenance_ballasting
    global zone_df_mechanised_resleepering
    global zone_df_mechanised_resurfacing
    zone_df_ballast_undercutting = zone_df_ballast_undercutting_
    zone_df_formation_repairs = zone_df_formation_repairs_
    zone_df_top_and_line_spot_resurfacing = zone_df_top_and_line_spot_resurfacing_
    zone_df_maintenance_ballasting = zone_df_maintenance_ballasting_
    zone_df_mechanised_resleepering = zone_df_mechanised_resleepering_
    zone_df_mechanised_resurfacing =  zone_df_mechanised_resurfacing_

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
    first_metrage=sorted_metrages[0]

    zone_df_ballast_undercutting = pd.DataFrame(0, columns=quarters, index=zones)
    zone_df_formation_repairs = pd.DataFrame(0, columns=quarters, index=zones)
    zone_df_top_and_line_spot_resurfacing = pd.DataFrame(0, columns=quarters, index=zones)
    zone_df_mechanised_resurfacing = pd.DataFrame(0, columns=quarters, index=zones)
    zone_df_maintenance_ballasting = pd.DataFrame(0, columns=quarters, index=zones)
    zone_df_mechanised_resleepering = pd.DataFrame(0, columns=quarters, index=zones)

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
    pool = ctx.Pool(n_jobs,initializer=init,initargs=(zone_df_ballast_undercutting,
                                                      zone_df_formation_repairs,     zone_df_top_and_line_spot_resurfacing,
                                                        zone_df_mechanised_resurfacing,
                                                      zone_df_maintenance_ballasting,
                                                      zone_df_mechanised_resleepering,
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