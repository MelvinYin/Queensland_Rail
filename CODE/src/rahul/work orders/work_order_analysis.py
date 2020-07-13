import pandas as pd
from os import listdir
from os.path import isfile, join
from CODE.src.utils import paths
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import pmdarima as pm
from sklearn.ensemble import RandomForestRegressor
import scipy.stats

RESPONSE_VARIABLE = "topL"
LINE="C138"
NUM_ZONES = "100"
WORK_ORDER_TYPES = ["Ballast Undercutting","Formation Repairs","Top & line Spot Resurfacing","Mechanised Resurfacing","Maintenance Ballasting","Mechanised Resleepering","total"]
# WORK_ORDER_TYPES = ["total"]
# DATA_DIRECTORY = paths.TRC_ML+"/work_order/"+str(LINE)+"/"+str(WORK_ORDER_TYPE)+"/"+str(NUM_ZONES)
# work_orders_zonal = pd.read_pickle(DATA_FILE_PATH)
# print(work_orders_zonal)

def load_data(dir):

    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    final_dataframe = pd.DataFrame
    for filename in onlyfiles:
        temp_data = pd.read_pickle(join(dir,filename))
        if final_dataframe.empty:
            final_dataframe = temp_data
        else:
            final_dataframe = final_dataframe.add(temp_data,fill_value=0)
    return final_dataframe


def plot_data(data,title):
    plt.pcolor(data, cmap=cm.Blues)
    plt.yticks(np.arange(0.5, len(data.index), 5), data.index[::5])
    plt.xticks(np.arange(0.5, len(data.columns), 5), data.columns[::5])
    plt.title(title+" "+LINE)
    plt.show()
    # plt.savefig("{}/plots/work_orders/{} {} {} zones.png".format(paths.REPORTS,LINE,title,NUM_ZONES))

def test_arima(data):
    model = pm.auto_arima(data.loc['zone1'].values, start_p=1, start_q=1,
                          test='adf',  # use adftest to find optimal 'd'
                          max_p=3, max_q=3,  # maximum p and q
                          m=1,  # frequency of series
                          d=None,  # let model determine 'd'
                          seasonal=False,  # No Seasonality
                          start_P=0,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
    print(model.summary())
    model.plot_diagnostics(figsize=(7, 5))

    plt.show()

def random_forest_regressor(X_train,X_test,y_train,y_test):
    num_trees = 10
    random_state = 1234
    # Instantiate model with num_trees decision trees
    RFmodel = RandomForestRegressor(n_estimators=num_trees, oob_score=True, random_state=random_state)

    # Train the model on training data
    print("training RandomForestRegressor")

    # X_train = np.delete(X_train, np.where(y_train == 0), axis=0)
    # y_train = np.delete(y_train, np.where(y_train == 0))

    X_test = np.delete(X_test, np.where(y_test == 0), axis=0)
    y_test = np.delete(y_test, np.where(y_test == 0))
    RFmodel.fit(X_train, y_train)
    # y_test = y_test.add(0.000001)

    # Generate predictions to assess performance
    RF_pred = RFmodel.predict(X_test)
    # RF_pred = np.delete(RF_pred, np.where(y_test == 0))


    # Calculate the absolute errors
    RFerrors = abs(RF_pred - y_test)

    # Calculate mean absolute percentage error (MAPE)
    RFmape = 100 * (RFerrors / y_test)
    RFaccuracy = 100 - np.mean(RFmape)
    print('RF accuracy (unscaled features):', round(RFaccuracy, 2), '%.')
    RFtest_score = RFmodel.score(X_test, y_test)
    print('RF test score:', round(RFtest_score, 3))
    print('Out of bag score:', round(RFmodel.oob_score_, 4))

    plt.scatter(RF_pred, y_test)
    plt.title('Linear Regression (unscaled): predicted vs. actual', fontsize=12)
    plt.xlabel('Predicted (test)', fontsize=10)
    plt.ylabel('Actual (test)', fontsize=10)

    # Plot line of best fit
    plt.plot(np.unique(RF_pred), np.poly1d(np.polyfit(RF_pred, y_test, 1))(np.unique(RF_pred)), color='red')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(RF_pred, y_test)
    M_string = 'M = ' + str(round(slope, 2)) + ', R = ' + str(round(r_value, 2))  # Extract gradient ~1.0
    plt.text(4, 7, M_string, color='red')
    plt.show()
    return RFmodel

def preprocess_data(data):

    num_points = data.shape[0]*data.shape[1]
    x = np.ndarray((0,0))
    y = np.ndarray((0,0))
    for zone,row in data.iterrows():
        for quarter, value in row.items():
            if x.shape[0]==0:
                x = np.array([int(quarter.split("quarter")[1])%4,int(zone.split("zone")[1])])
            else:
                x = np.vstack((x,[int(quarter.split("quarter")[1])%4,int(zone.split("zone")[1])]))
            if y.shape[0]==0:
                y = np.array([value])
            else:
                y = np.append(y,value)
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=1234)
    return x_train,x_test,y_train,y_test

def display_predictions(model,num_quarters):
    y = pd.DataFrame(columns=[i for i in range(23,23+num_quarters)],index=[i for i in range(1,101)])
    x =np.ndarray((0,0))
    for zone in range(1,101):
        for quarter in range(23,23+num_quarters):
            if x.shape[0]==0:
                x = np.array([quarter%4,zone])
            else:
                x = np.vstack((x,[quarter%4,zone]))
    y_pred = model.predict(x)
    index = 0
    for zone in range(1,101):
        for quarter in range(23,23+num_quarters):
            y.loc[zone,quarter] = int(y_pred[index])
            index +=1
    plt.pcolor(y, cmap=cm.Blues)
    plt.show()

def main():

    for work_order in WORK_ORDER_TYPES:
        dir = paths.TRC_ML + "/work_order/" + str(LINE) + "/" + work_order + "/" + str(NUM_ZONES)

        data = load_data(dir)
        # bins = np.array([1, 5, 25, 50, 150, 250, 1000, 5000, 10000])
        # x_train,x_test,y_train,y_test = preprocess_data(data)
        # model = random_forest_regressor(x_train,x_test,y_train,y_test)
        # file = open(paths.MODELS+'/rf_maintenance_zonal_'+LINE+'.pickle', 'wb')
        # pickle.dump(model,file)
        # display_predictions(model,num_quarters=4)

        # test_arima(data)
        for i in range(1,101):
            plt.plot(data.loc['zone'+str(i)])
            plt.xticks(np.arange(0.5, len(data.columns), 5), data.columns[::5])
        plt.show()
        plot_data(data,work_order)

if __name__=="__main__":
    main()