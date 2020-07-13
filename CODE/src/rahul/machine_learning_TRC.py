import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from CODE.src.utils import paths
import scipy.stats
import statsmodels.api as sms
from sklearn.ensemble import RandomForestRegressor
from os.path import isfile, join
from os import listdir
from joblib import dump, load

from sklearn.svm import SVR

# KNN
from sklearn import neighbors
from sklearn.metrics import mean_squared_error

# ANN Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
# ANN MLP
from sklearn.neural_network import MLPRegressor

#constants
metric = "combined"
RESPONSE_VARIABLE = "combined"
LINE="C138"
WITH_OR_WITHOUT_NULL="with"
DATA_FILE_PATH = paths.TRC_ML + "/" + LINE + " " + RESPONSE_VARIABLE + " " + WITH_OR_WITHOUT_NULL + " null.pkl"

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


def preprocess_data(data):
    dates = data.columns.values
    sorted_dates = sorted(dates, key=lambda d: tuple(map(int, d.split('-'))))
    test_proportion = 0.25
    nSamples = data.shape[0]
    indices = range(nSamples)
    random_state = 1234
    response_variable = sorted_dates[-2]
    # data = data.drop(response_variable,axis=1)
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
    return X_train,X_test,y_train,y_test,X_train_scaled,X_test_scaled,X_scaled,X,y


def get_p_values(X_train):
    # Rerun using OLS to obtain P-values
    X_train_features = sms.add_constant(X_train)  # Ensure an intercept is provided
    OLSmodel = sms.OLS(y_train, X_train)
    OLSresults = OLSmodel.fit()
    print(OLSresults.summary())

def linear_regressor_scaled(X_train_scaled,y_train,X_test_scaled,y_test):
    # Create linear regression model
    LRmodel_scaled = LinearRegression(fit_intercept=True, normalize=True)
    LRmodel_scaled.fit(X_train_scaled, y_train)
    X_train_scaled = np.delete(X_train_scaled.values, np.where(y_train<= 4.9), axis=0)
    y_train = np.delete(y_train.values, np.where(y_train<= 4.9))

    X_test_scaled = np.delete(X_test_scaled.values, np.where(y_test<= 4.9), axis=0)
    y_test = np.delete(y_test.values, np.where(y_test<= 4.9))

    r2 = LRmodel_scaled.score(X_train_scaled, y_train)
    print('scaled LR coefficient of determination:', round(r2, 4))

    # Generate predictions and assess accuracy
    # Generate predictions and assess accuracy
    LR_pred_scaled = LRmodel_scaled.predict(X_test_scaled)
    LRerrors = abs(LR_pred_scaled - y_test)

    # Calculate mean absolute percentage error (MAPE)
    # LRmape = 100 * np.delete((LRerrors / y_test), np.where(abs(LRerrors / y_test) > 100))
    LRmape = 100 * (LRerrors / abs(y_test))
    LRaccuracy = 100 - np.mean(LRmape)
    print('LR accuracy (scaled features):', round(LRaccuracy, 2), '%.')
    plt.scatter(LR_pred_scaled, y_test)
    plt.title('Linear Regression (scaled): predicted vs. actual', fontsize=12)
    plt.xlabel('Predicted (test)', fontsize=10)
    plt.ylabel('Actual (test)', fontsize=10)

    # Plot line of best fit
    plt.plot(np.unique(LR_pred_scaled), np.poly1d(np.polyfit(LR_pred_scaled, y_test, 1))(np.unique(LR_pred_scaled)), color='red')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(LR_pred_scaled, y_test)
    M_string = 'M = ' + str(round(slope, 2)) + ', R = ' + str(round(r_value, 2))  # Extract gradient ~1.0
    plt.text(4, 7, M_string, color='red')
    plt.show()



def linear_regressor_unscaled(X_train,X_test,y_train,y_test):
    # Create linear regression model
    LRmodel = LinearRegression(fit_intercept=True, normalize=False)
    LRmodel.fit(X_train, y_train)
    dump(LRmodel,"models/linear regression trained on "+RESPONSE_VARIABLE+ " for "+LINE+" line")
    X_train = np.delete(X_train.values, np.where(y_train <= 4.9), axis=0)
    y_train = np.delete(y_train.values, np.where(y_train <= 4.9))

    X_test = np.delete(X_test.values, np.where(abs(y_test)<= 4.9), axis=0)
    y_test = np.delete(y_test.values, np.where(abs(y_test)<= 4.9))

    r2 = LRmodel.score(X_train, y_train)
    print('unscaled Coefficient of determination:', round(r2, 4))

    # Generate predictions and assess accuracy
    LR_pred = LRmodel.predict(X_test)
    LRerrors = abs(LR_pred - y_test)

    # Calculate mean absolute percentage error (MAPE)
    # LRmape = 100 * np.delete((LRerrors / y_test),np.where(abs(LRerrors / y_test)>100))
    LRmape = 100 * abs(LRerrors / y_test)
    LRaccuracy = 100 - np.mean(LRmape)
    print('LR accuracy (unscaled features):', round(LRaccuracy, 2), '%.')

    plt.scatter(LR_pred, y_test)
    plt.title('Linear Regression (unscaled): predicted vs. actual', fontsize=12)
    plt.xlabel('Predicted (test)', fontsize=10)
    plt.ylabel('Actual (test)', fontsize=10)

    # Plot line of best fit
    plt.plot(np.unique(LR_pred), np.poly1d(np.polyfit(LR_pred, y_test, 1))(np.unique(LR_pred)), color='red')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(LR_pred, y_test)
    M_string = 'M = ' + str(round(slope, 2)) + ', R = ' + str(round(r_value, 2))  # Extract gradient ~1.0
    plt.text(4, 7, M_string, color='red')
    plt.show()

def load_and_run_linear_regressor_unscaled(filepath,X_test,y_test):
    # Create linear regression model
    LRmodel = load(filepath)
    # dump(LRmodel,"models/linear regression trained on "+RESPONSE_VARIABLE+ " for "+LINE+" line")


    # r2 = LRmodel.score(X_train, y_train)
    # print('unscaled Coefficient of determination:', round(r2, 4))

    # Generate predictions and assess accuracy
    LR_pred = LRmodel.predict(X_test)
    LRerrors = abs(LR_pred - y_test)

    # Calculate mean absolute percentage error (MAPE)
    LRmape = 100 * (LRerrors / y_test).replace([np.inf, -np.inf])
    LRaccuracy = 100 - np.mean(LRmape)
    print('unscaled LR accuracy:', round(LRaccuracy, 2), '%.')

    plt.scatter(LR_pred, y_test)
    plt.title('Linear Regression (unscaled): predicted vs. actual', fontsize=12)
    plt.xlabel('Predicted (test)', fontsize=10)
    plt.ylabel('Actual (test)', fontsize=10)

    # Plot line of best fit
    plt.plot(np.unique(LR_pred), np.poly1d(np.polyfit(LR_pred, y_test, 1))(np.unique(LR_pred)), color='red')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(LR_pred, y_test)
    M_string = 'M = ' + str(round(slope, 2)) + ', R = ' + str(round(r_value, 2))  # Extract gradient ~1.0
    plt.text(4, 7, M_string, color='red')
    plt.show()


def keras_ann(X_train_scaled,X_test_scaled,y_train,y_test):
    # X_train_scaled = np.delete(X_train_scaled.values, np.where(y_train <= 4.9), axis=0)
    # y_train = np.delete(y_train.values, np.where(y_train <= 4.9))

    X_test_scaled = np.delete(X_test_scaled.values, np.where(abs(y_test) <= 4.9), axis=0)
    y_test = np.delete(y_test.values, np.where(abs(y_test) <= 4.9))
    # Key variables
    n_epochs = 10  # iterations
    ANNbatch = 100  # batches of training samples to propogate to optimise memory
    ANN_CV = 10  # kfold CV

    # Specify the keras model
    def ANNbase_model():
        ANNmodel = Sequential()
        ANNmodel.add(Dense(16, input_dim=X_train_scaled.shape[1], activation='relu'))
        # ANNmodel.add(Dense(64, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
        # ANNmodel.add(Dense(32, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
        # ANNmodel.add(Dense(16, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
        ANNmodel.add(Dense(8, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
        ANNmodel.add(Dense(4, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
        ANNmodel.add(Dense(1))

        # compile the keras model
        ANNmodel.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['accuracy'])

        return ANNmodel

    # Evalate ANN
    ANNestimator = KerasRegressor(build_fn=ANNbase_model, epochs=n_epochs, batch_size=ANNbatch, verbose=1)
    # ANNkfold = KFold(n_splits=ANN_CV)
    # ANNresults = cross_val_score(ANNestimator, X_scaled, y, cv=ANNkfold)
    # print("Baseline: %.2f (%.2f) MSE" % (ANNresults.mean(), ANNresults.std()))

    # Evaluate test acuracy
    ANNestimator.fit(X_train_scaled, y_train)
    ANN_pred_35 = ANNestimator.predict(X_test_scaled)
    ANNerrors_35 = abs(ANN_pred_35 - y_test)

    # Calculate mean absolute percentage error (MAPE)
    ANNmape_35 = 100 * (ANNerrors_35 / y_test)
    ANNaccuracy_35 = 100 - np.mean(ANNmape_35)
    print('ANN accuracy (features=', X_train_scaled.shape[1], '):', round(ANNaccuracy_35, 2), '%.')
    ANNtest_score_35 = ANNestimator.score(X_test_scaled, y_test)
    print('ANN test score:', round(ANNtest_score_35, 3))

    plt.scatter(ANN_pred_35, y_test)
    plt.title('ANN Regression: predicted vs. actual (delta feature)', fontsize=12)
    plt.xlabel('Predicted (test)', fontsize=10)
    plt.ylabel('Actual (test)', fontsize=10)

    # Plot line of best fit
    plt.plot(np.unique(ANN_pred_35), np.poly1d(np.polyfit(ANN_pred_35, y_test, 1))(np.unique(ANN_pred_35)), color='red')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(ANN_pred_35, y_test)
    M_string = 'M = ' + str(round(slope, 2)) + ', R = ' + str(round(r_value, 2))  # Extract gradient ~1.0
    plt.text(4, 7, M_string, color='red')
    plt.show()

def mlp_regressor(X_train_scaled,X_test_scaled,y_train,y_test):
    # X_train_scaled = np.delete(X_train_scaled.values, np.where(y_train <= 4.9), axis=0)
    # y_train = np.delete(y_train.values, np.where(y_train <= 4.9))

    X_test_scaled = np.delete(X_test_scaled.values, np.where(abs(y_test) <= 4.9), axis=0)
    y_test = np.delete(y_test.values, np.where(abs(y_test) <= 4.9))
    # Define MLP
    MLPbatch = int(X_train_scaled.shape[0] / 2)
    MLPmodel_35 = MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32, 16, 8, 4, 1), activation='relu', solver='lbfgs',
                               alpha=1e-2, batch_size=MLPbatch, learning_rate='invscaling', learning_rate_init=1e-6,
                               power_t=0.5, max_iter=10000, shuffle=True, random_state=None, tol=0.0001,
                               verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                               early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                               epsilon=1e-10, n_iter_no_change=50)

    MLPmodel_35.fit(X_train_scaled, y_train)
    MLP_pred_35 = MLPmodel_35.predict(X_test_scaled)

    MLPerrors_35 = abs(MLP_pred_35 - y_test)

    # Calculate mean absolute percentage error (MAPE)
    MLPmape_35 = 100 * abs(MLPerrors_35 / y_test)
    MLPaccuracy_35 = 100 - np.mean(MLPmape_35)
    print('MLP accuracy (features=', X_train_scaled.shape[1], '):', round(MLPaccuracy_35, 2), '%.')
    MLPtest_score_35 = MLPmodel_35.score(X_test_scaled, y_test)
    print('MLP test score:', round(MLPtest_score_35, 3))

    plt.scatter(MLP_pred_35, y_test)
    plt.title('MLP Regression: predicted vs. actual (delta feature)', fontsize=12)
    plt.xlabel('Predicted (test)', fontsize=10)
    plt.ylabel('Actual (test)', fontsize=10)

    # Plot line of best fit
    plt.plot(np.unique(MLP_pred_35), np.poly1d(np.polyfit(MLP_pred_35, y_test, 1))(np.unique(MLP_pred_35)), color='red')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(MLP_pred_35, y_test)
    M_string = 'M = ' + str(round(slope, 2)) + ', R = ' + str(round(r_value, 2))  # Extract gradient ~1.0
    plt.text(4, 7, M_string, color='red')
    plt.show()

def svr(X_train_scaled,X_test_scaled,y_train,y_test):
    X_train_scaled = np.delete(X_train_scaled.values, np.where(y_train <= 4.9), axis=0)
    y_train = np.delete(y_train.values, np.where(y_train <= 4.9))

    X_test_scaled = np.delete(X_test_scaled.values, np.where(abs(y_test) <= 4.9), axis=0)
    y_test = np.delete(y_test.values, np.where(abs(y_test) <= 4.9))
    # Vary C to optimise SVR NB C=0 = no penalty
    C_array = [0.01, 0.1, 1.0, 10, 100, 1000]
    SVR_array = []  # array will store C, test accuracy and test score

    for C_iter in C_array:
        print('\nC:', C_iter)

        SVRmodel = SVR(C=C_iter, kernel='rbf', gamma='auto', epsilon=0.01)
        SVRmodel.fit(X_train_scaled, y_train)

        # Generate predictions to assess performance
        SVR_pred = SVRmodel.predict(X_test_scaled)  # Calculate the absolute errors
        SVRerrors = abs(SVR_pred - y_test)

        # Calculate mean absolute percentage error (MAPE)
        SVRmape = 100 * (SVRerrors / abs(y_test))
        SVRaccuracy = 100 - np.mean(SVRmape)
        print('SVR accuracy:', round(SVRaccuracy, 2), '%.')
        SVRtest_score = SVRmodel.score(X_test_scaled, y_test)
        print('SVR test score:', round(SVRtest_score, 3))
        SVR_array.append([C_iter, SVRaccuracy, SVRtest_score])

    SVR_array = np.asarray(SVR_array)

    # Refit using optimal params
    SVR_best_C = 10
    SVR_best_epsilon = 0.01

    SVRmodel = SVR(C=SVR_best_C, kernel='rbf', gamma='auto', epsilon=SVR_best_epsilon)
    SVRmodel.fit(X_train_scaled, y_train)

    # Generate predictions to plot predicted and actuals
    SVR_pred = SVRmodel.predict(X_test_scaled)

    plt.scatter(SVR_pred, y_test)
    plt.title('Support Vector Regression: predicted vs. actual', fontsize=12)
    plt.xlabel('Predicted (test)', fontsize=10)
    plt.ylabel('Actual (test)', fontsize=10)

    # Plot line of best fit
    plt.plot(np.unique(SVR_pred), np.poly1d(np.polyfit(SVR_pred, y_test, 1))(np.unique(SVR_pred)), color='red')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(SVR_pred, y_test)
    M_string = 'M = ' + str(round(slope, 2)) + ', R = ' + str(round(r_value, 2))  # Extract gradient ~1.0
    plt.text(4, 7, M_string, color='red')
    plt.show()

    # Vary C to optimise SVR NB C=0 = no penalty
    C_array = [0.01, 0.1, 1.0, 10, 100, 1000]
    C_array = [0.01, 0.1, 1.0, 10, 100, 1000]
    SVR_array = []  # array will store C, test accuracy and test score

    for C_iter in C_array:
        print('\nC:', C_iter)

        SVRmodel = SVR(C=C_iter, kernel='sigmoid', gamma='auto', epsilon=0.01)
        SVRmodel.fit(X_train_scaled, y_train)

        # Generate predictions to assess performance
        SVR_pred = SVRmodel.predict(X_test_scaled)  # Calculate the absolute errors
        SVRerrors = abs(SVR_pred - y_test)

        # Calculate mean absolute percentage error (MAPE)
        SVRmape = 100 * (SVRerrors / abs(y_test))
        SVRaccuracy = 100 - np.mean(SVRmape)
        print('SVR accuracy sigmoid kernel:', round(SVRaccuracy, 2), '%.')
        SVRtest_score = SVRmodel.score(X_test_scaled, y_test)
        print('SVR test score sigmoid kernel :', round(SVRtest_score, 3))
        SVR_array.append([C_iter, SVRaccuracy, SVRtest_score])

    SVR_array = np.asarray(SVR_array)

    # Vary C to optimise SVR NB C=0 = no penalty
    C_array = [0.01, 0.1, 1.0, 10, 100, 1000]
    SVR_array = []  # array will store C, test accuracy and test score

    for C_iter in C_array:
        print('\nC:', C_iter)

        SVRmodel = SVR(C=C_iter, kernel='poly', degree=3, gamma='auto', epsilon=0.1)
        SVRmodel.fit(X_train_scaled, y_train)

        # Generate predictions to assess performance
        SVR_pred = SVRmodel.predict(X_test_scaled)  # Calculate the absolute errors
        SVRerrors = abs(SVR_pred - y_test)

        # Calculate mean absolute percentage error (MAPE)
        SVRmape = 100 * (SVRerrors / abs(y_test))
        SVRaccuracy = 100 - np.mean(SVRmape)
        print('SVR accuracy poly kernel:', round(SVRaccuracy, 2), '%.')
        SVRtest_score = SVRmodel.score(X_test_scaled, y_test)
        print('SVR test score poly kernel:', round(SVRtest_score, 3))
        SVR_array.append([C_iter, SVRaccuracy, SVRtest_score])

    SVR_array = np.asarray(SVR_array)


def knn(X_train_scaled,X_test_scaled,y_train,y_test):
    KNNrmse_array = []  # to store rmse values for different k
    K_min = 5  # Min K value
    K_max = 250  # Max K value
    K_inc = 5  # Increment in K search

    for K in range(K_min, K_max, K_inc):
        KNNmodel_35 = neighbors.KNeighborsRegressor(n_neighbors=K)

        KNNmodel_35.fit(X_train_scaled, y_train)  # fit the model
        KNNpred_35 = KNNmodel_35.predict(X_test_scaled)  # make prediction on test set
        KNNrmse = (mean_squared_error(y_test, KNNpred_35)) ** 0.5  # calculate rmse
        KNNrmse_array.append([K, KNNrmse])  # store rmse values

    # Plot RMSE vs. K
    KNNelbow = pd.DataFrame(KNNrmse_array)  # elbow curve
    plt.plot(KNNelbow.loc[:, 0], KNNelbow.loc[:, 1], '.b-')
    plt.title('KNN Test eror vs. K')
    plt.xlabel('K')
    plt.ylabel('Test RMSE')
    ax = plt.gca()
    plt.show()

    # Extract optimal K
    KNN_opt_ix = KNNelbow.loc[:, 1].idxmin(axis=1)
    K_opt = KNNelbow.iloc[KNN_opt_ix, 0]

    # Refit using optimal K
    KNNmodel_35 = neighbors.KNeighborsRegressor(n_neighbors=K_opt)

    KNNmodel_35.fit(X_train_scaled, y_train)  # fit the model

    # Generate predictions to plot predicted and actuals
    KNN_pred_35 = KNNmodel_35.predict(X_test_scaled)
    KNNerrors_35 = abs(KNN_pred_35 - y_test)

    # Calculate mean absolute percentage error (MAPE)
    KNNmape_35 = 100 * (KNNerrors_35 / y_test)
    KNNaccuracy_35 = 100 - np.mean(KNNmape_35)
    print('Optimal K:', K_opt)
    print('KNN accuracy (features=', X_train_scaled.shape[1], '):', round(KNNaccuracy_35, 2), '%.')
    KNNtest_score_35 = KNNmodel_35.score(X_test_scaled, y_test)
    print('KNN test score:', round(KNNtest_score_35, 3))

    plt.scatter(KNN_pred_35, y_test)
    plt.title('KNN Regression: predicted vs. actual (delta feature)', fontsize=12)
    plt.xlabel('Predicted (test)', fontsize=10)
    plt.ylabel('Actual (test)', fontsize=10)

    # Plot line of best fit
    plt.plot(np.unique(KNN_pred_35), np.poly1d(np.polyfit(KNN_pred_35, y_test, 1))(np.unique(KNN_pred_35)), color='red')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(KNN_pred_35, y_test)
    M_string = 'M = ' + str(round(slope, 2)) + ', R = ' + str(round(r_value, 2))  # Extract gradient ~1.0
    plt.text(4, 7, M_string, color='red')
    plt.show()

def random_forest_regressor(X_train,X_test,y_train,y_test):
    X_train = np.delete(X_train.values, np.where(y_train <= 4.9), axis=0)
    y_train = np.delete(y_train.values, np.where(y_train <= 4.9))

    X_test = np.delete(X_test.values, np.where((y_test <= 4.9 )| (y_test >=10)), axis=0)
    y_test = np.delete(y_test.values, np.where((y_test <= 4.9 )| (y_test >=10)))

    # X_test = np.delete(X_test.values, np.where(y_test >=10), axis=0)
    # y_test = np.delete(y_test.values, np.where(y_test >=10))


    num_trees = 100
    random_state = 1234
    # Instantiate model with num_trees decision trees
    RFmodel = RandomForestRegressor(n_estimators=num_trees, oob_score=True, random_state=random_state)

    # Train the model on training data
    print("training RandomForestRegressor")

    # X_train = np.delete(X_train.values, np.where(y_train == 0), axis=0)
    # y_train = np.delete(y_train.values, np.where(y_train == 0))

    # X_test = np.delete(X_test, np.where(y_test == 0), axis=0)
    # y_test = np.delete(y_test.values, np.where(y_test == 0))
    RFmodel.fit(X_train, y_train)
    # y_test = y_test.add(0.000001)

    # Generate predictions to assess performance
    RF_pred = RFmodel.predict(X_test)
    # RF_pred = np.delete(RF_pred, np.where(y_test == 0))


    # Calculate the absolute errors
    RFerrors = abs(RF_pred - y_test)

    # Calculate mean absolute percentage error (MAPE)
    # RFmape = 100 * np.delete((RFerrors / y_test),np.where(abs(RFerrors / y_test)>100))
    RFmape = 100 * abs(RFerrors / y_test)
    RFaccuracy = 100 - np.mean(RFmape)
    print('RF accuracy (unscaled features):', round(RFaccuracy, 2), '%.')
    RFtest_score = RFmodel.score(X_test, y_test)
    print('coefficient of determination R^2:', round(RFtest_score, 3))
    print('Out of bag score:', round(RFmodel.oob_score_, 4))

    plt.scatter(RF_pred, y_test)
    plt.title('Random Forest Regression: predicted vs. actual', fontsize=12)
    plt.xlabel('Predicted (test)', fontsize=10)
    plt.ylabel('Actual (test)', fontsize=10)

    # Plot line of best fit
    plt.plot(np.unique(RF_pred), np.poly1d(np.polyfit(RF_pred, y_test, 1))(np.unique(RF_pred)), color='red')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(RF_pred, y_test)
    M_string = 'M = ' + str(round(slope, 2)) + ', R = ' + str(round(r_value, 2))  # Extract gradient ~1.0
    plt.text(4, 7, M_string, color='red')
    plt.show()




if __name__=="__main__":
    filepath = paths.TRC_ML + "/trc/" + LINE + "/" + metric

    data = load_data(filepath)

    x_train,\
    x_test,\
    y_train,\
    y_test,\
    x_train_scaled,\
    x_test_scaled,\
    x_scaled,\
    x,\
    y= preprocess_data(data)
    # linear_regressor_unscaled(x_train,x_test,y_train,y_test)
    random_forest_regressor(x_train,x_test,y_train,y_test)
    # linear_regressor_scaled(x_train_scaled, y_train,x_test_scaled, y_test)
    # keras_ann(x_train_scaled, x_test_scaled, y_train, y_test)
    # mlp_regressor(x_train_scaled, x_test_scaled, y_train, y_test)
    # filepath="models/linear regression trained on "+RESPONSE_VARIABLE+ " for C138 line"
    # load_and_run_linear_regressor_unscaled(filepath,x_test,y_test)

    # svr(x_train_scaled, x_test_scaled, y_train, y_test)




















