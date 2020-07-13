import pandas as pd
from os import listdir
from os.path import isfile, join
from CODE.src.utils import paths
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

RESPONSE_VARIABLE = "topL"
LINE="C195"
NUM_ZONES = "100"
metrics = ["combined","topl","topr","twist3"]
# DATA_DIRECTORY = paths.TRC_ML+"/work_order/"+str(LINE)+"/"+str(WORK_ORDER_TYPE)+"/"+str(NUM_ZONES)
# work_orders_zonal = pd.read_pickle(DATA_FILE_PATH)
# print(work_orders_zonal)

def load_data(dir):

    onlyfiles_total = [f for f in listdir(dir) if isfile(join(dir, f))]
    final_dataframe = pd.DataFrame
    for i,filename in enumerate(onlyfiles_total):
        temp_data = pd.read_pickle(join(dir,filename))
        if final_dataframe.empty:
            final_dataframe = temp_data
        else:
            final_dataframe = final_dataframe.add(temp_data,fill_value=0)



    return final_dataframe


def plot_data(data,title="combined"):
    plt.pcolor(data, cmap=cm.Blues)
    plt.yticks(np.arange(0.5, len(data.index), 5), data.index[::5])
    plt.xticks(np.arange(0.5, len(data.columns), 5), data.columns[::5])
    plt.title(title+" "+LINE)
    plt.draw()
    plt.savefig("{}/plots/trc_zonal/{} {} {} zones linear.png".format(paths.REPORTS, LINE, title, NUM_ZONES))

def main():

        for metric in metrics :

            dir = paths.TRC_ML + "/trc/" + str(LINE) + "/" + metric + "_zonal_linear/"

            data = load_data(dir)
            plot_data(data,metric)

if __name__=="__main__":
    main()