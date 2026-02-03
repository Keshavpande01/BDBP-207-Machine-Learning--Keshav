
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

X = data.iloc[:, 0:5].values   # features
y = data.iloc[:, 5].values    # target

# 2. TRAIN–TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=999
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))


m, n = X_train.shape
thetas = np.zeros(n)


alpha = 0.01
iterations = 2000

for i in range(iterations):
    y_pred = X_train @ thetas
    error = y_pred - y_train

    gradient = (1 / m) * (X_train.T @ error)
    thetas = thetas - alpha * gradient


y_train_pred = X_train @ thetas
y_test_pred = X_test @ thetas


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


print("Train MSE :", mse(y_train, y_train_pred))
print("Test  MSE :", mse(y_test, y_test_pred))

print("Train R²  :", r2_score(y_train, y_train_pred))
print("Test  R²  :", r2_score(y_test, y_test_pred))

print("Final Thetas:", thetas)
