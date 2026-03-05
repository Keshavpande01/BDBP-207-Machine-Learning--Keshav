import numpy as np
import pandas as pd

data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

X = data.iloc[:, 0:5].values
y = data.iloc[:, 6].values


def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2):
    n = len(X)
    indices = np.random.permutation(n)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])

def add_intercept(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def normal_equation(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def feature_selection(X_train, y_train, X_val, y_val):

    n_features = X_train.shape[1]
    best_mse = float('inf')
    best_features = None

    for i in range(n_features):

        # Drop feature i
        X_train_sub = np.delete(X_train, i, axis=1)
        X_val_sub = np.delete(X_val, i, axis=1)

        X_train_sub = add_intercept(X_train_sub)
        X_val_sub = add_intercept(X_val_sub)

        theta = normal_equation(X_train_sub, y_train)
        y_pred = X_val_sub @ theta

        val_mse = mse(y_val, y_pred)

        print(f"Dropping feature {i}, Val MSE = {val_mse:.4f}")

        if val_mse < best_mse:
            best_mse = val_mse
            best_features = i

    return best_features, best_mse