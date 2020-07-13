from os import listdir
from os.path import isfile, join
from CODE.src.utils import paths
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from src.utils import paths
import scipy.stats

from joblib import dump

# KNN
metric = "pvc_combined"
LINE = "C138"

def load_data(dir):


    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    final_dataframe = pd.DataFrame()
    for filename in onlyfiles:
        try:
            temp_data = pd.read_pickle(join(dir,filename)).fillna(0.0).astype(np.float64)
            if final_dataframe.empty:
                final_dataframe = temp_data
            else:
                final_dataframe = final_dataframe.add(temp_data,fill_value=0)
        except Exception as e:
            print(e)
    # final_dataframe = final_dataframe[(final_dataframe[final_dataframe.columns] >= 0).all(axis=1)]
    return final_dataframe

def get_metric_to_column_mapping():

    df = pd.read_excel(paths.DATA + "/Mapping GPR Column Names to GPR Features.xlsx")
    metric_to_column = dict()
    for index,row in df.iterrows():
            metric_to_column[row["GPR Feature"]] = row["GPR Column Name"]
            
    return metric_to_column


def preprocess_data(data):
    dates = data.columns.values
    sorted_dates = sorted(dates, key=lambda d: tuple(map(int, d.split('-'))))
    test_proportion = 0.25
    nSamples = data.shape[0]
    indices = range(nSamples)
    random_state = 1234
    response_variable = sorted_dates[-1]
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(data,
                                                                                     data[response_variable],
                                                                                     indices, test_size=test_proportion,
                                                                                     random_state=random_state)

    # print(X_test)

    X_train = X_train.drop(response_variable, axis=1)
    X_test = X_test.drop(response_variable, axis=1)
    nFeatures = X_train.shape[1]  # number of features

    # For use in cross validation - do not split training and test
    X = data.drop(response_variable, axis=1)
    y = data[response_variable]

    # Get mean of y_test for model assessment
    y_test_mean = np.mean(y_test)

    # Normalise features
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = scaler.fit_transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns)
    return X_train,X_test,y_train,y_test,X_train_scaled,X_test_scaled,X_scaled

def linear_regressor_unscaled(X_train,X_test,y_train,y_test):
    # Create linear regression model
    LRmodel = LinearRegression(fit_intercept=True, normalize=False)
    LRmodel.fit(X_train, y_train)
    filename = paths.SRC + "/rahul/models/gpr/" + metric + " for " + LINE + " linear regression.model"
    f = open(filename,"w+")
    dump(LRmodel,filename)


    r2 = LRmodel.score(X_train, y_train)
    print('unscaled Coefficient of determination:', round(r2, 4))

    # Generate predictions and assess accuracy
    LR_pred = LRmodel.predict(X_test)
    LRerrors = abs(LR_pred - y_test)

    # Calculate mean absolute percentage error (MAPE)
    LRmape = 100 * (LRerrors / y_test).replace([np.inf, -np.inf])
    LRaccuracy = 100 - np.mean(LRmape)
    print('unscaled LR accuracy:', round(LRaccuracy, 2), '%.')

    plt.scatter(LR_pred, y_test)
    plt.title(metric+' Linear Regression (unscaled): predicted vs. actual', fontsize=12)
    plt.xlabel('Predicted (test)', fontsize=10)
    plt.ylabel('Actual (test)', fontsize=10)

    # Plot line of best fit
    plt.plot(np.unique(LR_pred), np.poly1d(np.polyfit(LR_pred, y_test, 1))(np.unique(LR_pred)), color='red')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(LR_pred, y_test)
    M_string = 'M = ' + str(round(slope, 2)) + ', R = ' + str(round(r_value, 2))  # Extract gradient ~1.0
    plt.text(4, 7, M_string, color='red')
    plt.show()


def main():
    # metric_to_column = get_metric_to_column_mapping()
    # # print(metric_to_column)

    line = "138"
    filepath = paths.TRC_ML + "/gpr/" + LINE + "/" + metric
    data = load_data(filepath).dropna()


    x_train, \
    x_test, \
    y_train, \
    y_test, \
    x_train_scaled, \
    x_test_scaled, \
    x_scaled = preprocess_data(data)
    linear_regressor_unscaled(x_train,x_test,y_train,y_test)






if __name__=="__main__":
    main()


