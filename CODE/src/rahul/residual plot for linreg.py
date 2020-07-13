import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from CODE.src.utils import paths

modified_df = pd.read_pickle(paths.TRC_ML + "/C138 topR with null.pkl").fillna(0.0)

dates = modified_df.columns.values
sorted_dates = sorted(dates, key=lambda d:tuple(map(int, d.split('-'))))


# modified_df = modified_df.loc[(modified_df['2019-02-07'] > -2.5) & (modified_df['2019-02-07'] < 2.5)]


test_proportion = 0.25
nSamples = modified_df.shape[0]
indices = range(nSamples)
random_state = 1234
response_variable = sorted_dates[-2]
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(modified_df, modified_df[response_variable],
                            indices, test_size=test_proportion, random_state = random_state)

# print(X_test)


X_train = X_train.drop(response_variable, axis=1)
X_test = X_test.drop(response_variable, axis=1)
nFeatures = X_train.shape[1] # number of features

# For use in cross validation - do not split training and test
X = modified_df.drop(response_variable, axis=1)
y = modified_df[response_variable]

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


# Create linear regression model
LRmodel = LinearRegression(fit_intercept = True, normalize = False)
LRmodel.fit(X_train, y_train)

r2 = LRmodel.score(X_train, y_train)
print('unscaled Coefficient of determination:', round(r2,4))

# Generate predictions and assess accuracy
LR_pred = LRmodel.predict(X_test)
LRerrors = abs(LR_pred - y_test)

# Calculate mean absolute percentage error (MAPE)
LRmape = 100 * (LRerrors / y_test).replace([np.inf, -np.inf])
LRaccuracy = 100 - np.mean(LRmape)
print('unscaled LR accuracy:', round(LRaccuracy, 2), '%.')

# Create linear regression model
LRmodel_scaled = LinearRegression(fit_intercept = True, normalize = True)
LRmodel_scaled.fit(X_train_scaled, y_train)

r2 = LRmodel_scaled.score(X_train_scaled, y_train)
print('scaled LR coefficient of determination:', round(r2,4))

# Generate predictions and assess accuracy
LR_pred_scaled = LRmodel_scaled.predict(X_test_scaled)
LRerrors = abs(LR_pred_scaled - y_test)

# Calculate mean absolute percentage error (MAPE)
LRmape = 100 * (LRerrors / y_test).replace([np.inf, -np.inf])
LRaccuracy = 100 - np.mean(LRmape)
print('scaled LR accuracy:', round(LRaccuracy, 2), '%.')

from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(LRmodel)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure