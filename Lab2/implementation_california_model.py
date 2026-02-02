
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import load_dataset

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

#def eda(X_df,X_df_scaled):
# 1)Load Data Set


# 1)Load Data Set
def load_data():
    [X,y] = fetch_california_housing(return_X_y=True)
    return X,y
X,y=load_data()

print(X.shape)
print(y.shape)

# 2) Divide it into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

def main():
    #3)Standardize the data

    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scale = scaler.transform(X_test)
    print(X_train_scaled.shape)
    print(X_test_scale.shape)

    #4 Intializing the model
    model = LinearRegression()

    #5 Training the model
    model.fit(X_train_scaled, y_train)

    #6 Test the
    y_pred = model.predict(X_test_scale)
    r2 = r2_score(y_test, y_pred)
    print(r2)

    #from sklearn.datasets import fetch_california_housing
    #california_housing = fetch_california_housing(as_frame=True)
    #print(california_housing.DESCR)
    print('Done !')

    plt.plot(X_test_scale, y_test, 'b')
    plt.show()

if __name__=='__main__':
    main()

