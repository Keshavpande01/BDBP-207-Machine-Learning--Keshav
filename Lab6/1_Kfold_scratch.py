"""
Simple K-Fold Cross Validation (Normal Equation vs Sklearn)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ==============================
# NORMAL EQUATION FUNCTION
# ==============================

def normal_equation(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y


# ==============================
# SCRATCH K-FOLD
# ==============================

def kfold_scratch(X, y, k=4):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    mse_list = []
    r2_list  = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        # Add intercept
        X_train = np.c_[np.ones(len(X_train)), X_train]
        X_val   = np.c_[np.ones(len(X_val)), X_val]

        theta = normal_equation(X_train, y_train)
        y_pred = X_val @ theta

        # theta = sgd_train(X_train, y_train)
        # y_pred = X_val @ theta

        mse = mean_squared_error(y_val, y_pred)
        r2  = r2_score(y_val, y_pred)

        mse_list.append(mse)
        r2_list.append(r2)

        print(f"Fold {fold} | MSE: {mse:.4f} | R²: {r2:.4f}")

    print("\nScratch Average MSE:", np.mean(mse_list))
    print("Scratch Average R² :", np.mean(r2_list))

    return np.mean(mse_list), np.mean(r2_list)


# ==============================
# SKLEARN K-FOLD
# ==============================

def kfold_sklearn(X, y, k=4):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    mse_list = []
    r2_list  = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        mse = mean_squared_error(y_val, y_pred)
        r2  = r2_score(y_val, y_pred)

        mse_list.append(mse)
        r2_list.append(r2)

        print(f"Fold {fold} | MSE: {mse:.4f} | R²: {r2:.4f}")

    print("\nSklearn Average MSE:", np.mean(mse_list))
    print("Sklearn Average R² :", np.mean(r2_list))

    return np.mean(mse_list), np.mean(r2_list)


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

    X = data.iloc[:, 0:5].values
    y = data.iloc[:, 6].values

    print("\n===== SCRATCH (Normal Equation) =====")
    scratch_mse, scratch_r2 = kfold_scratch(X, y)

    print("\n===== SKLEARN (LinearRegression) =====")
    sklearn_mse, sklearn_r2 = kfold_sklearn(X, y)

    print("\n===== COMPARISON =====")
    print(f"Scratch  -> MSE: {scratch_mse:.4f}, R²: {scratch_r2:.4f}")
    print(f"Sklearn  -> MSE: {sklearn_mse:.4f}, R²: {sklearn_r2:.4f}")