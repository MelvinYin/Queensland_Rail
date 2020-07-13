import pandas as pd
from CODE.src.utils import paths
import math

NUM_ZONES =10
WORK_ORDER_TYPE = "Mechanised Resurfacing"
PART = "1"
INPUT_FILE = "/Users/rahul.chowdhury/Downloads/TRC_joined_16nov/TRC_195_16nov/combined_195.csv"

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


if __name__ == "__main__":

    data = load_data_from_file(INPUT_FILE)
    dates = data["Date"].unique()
    sorted_dates = sorted(dates, key=lambda d: tuple(map(int, d.split('-'))))
    metrages = data["METRAGE"].unique()
    sorted_metrages = sorted(metrages)
    zone_length = get_zone_length(sorted_metrages)
    quarters = ["quarter"+str(i) for i in range(1,23)]
    zones = ["zone"+str(i) for i in range(1,11)]
    first_metrage = sorted_metrages[0]

    zone_df = pd.DataFrame(0,columns=quarters, index=zones)
    for i, row in data.iterrows():
        if WORK_ORDER_TYPE in row["Work_order_type"]:
            zone = "zone"+str(math.ceil((row['METRAGE']-first_metrage)/zone_length))
            quarter = "quarter"+str(get_quarter(row["Date"]))
            zone_df.loc[zone][quarter] +=1
    zone_df.to_pickle(paths.TRC_ML + "/C195 work order data for Mechanised Resurfacing with 10 zones.pkl")

