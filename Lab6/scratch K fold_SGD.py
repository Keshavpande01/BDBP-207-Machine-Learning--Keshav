#
#
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import KFold
# from sklearn.linear_model import LinearRegression, SGDRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
#
#
# # =====================================================
# # MODELS
# # =====================================================
#
# def normal_equation(X, y):
#     return np.linalg.pinv(X.T @ X) @ X.T @ y
#
#
# def sgd_train(X, y, lr=0.01, epochs=100):
#     theta = np.zeros(X.shape[1])
#
#     for _ in range(epochs):
#         for i in range(len(X)):
#             error = X[i] @ theta - y[i]
#             theta -= lr * X[i] * error
#
#     return theta
#
#
# # =====================================================
# # GENERAL K-FOLD FUNCTION
# # =====================================================
#
# def run_kfold(X, y, model_type="normal", k=10):
#
#     kf = KFold(n_splits=k, shuffle=True, random_state=42)
#     mse_list, r2_list = [], []
#
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
#
#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]
#
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_val   = scaler.transform(X_val)
#
#         # --------- SCRATCH NORMAL ----------
#         if model_type == "normal":
#             X_train_i = np.c_[np.ones(len(X_train)), X_train]
#             X_val_i   = np.c_[np.ones(len(X_val)), X_val]
#
#             theta = normal_equation(X_train_i, y_train)
#             y_pred = X_val_i @ theta
#
#         # --------- SCRATCH SGD ----------
#         elif model_type == "sgd":
#             X_train_i = np.c_[np.ones(len(X_train)), X_train]
#             X_val_i   = np.c_[np.ones(len(X_val)), X_val]
#
#             theta = sgd_train(X_train_i, y_train)
#             y_pred = X_val_i @ theta
#
#         # --------- SKLEARN NORMAL ----------
#         elif model_type == "sklearn_normal":
#             model = LinearRegression()
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_val)
#
#         # --------- SKLEARN SGD ----------
#         elif model_type == "sklearn_sgd":
#             model = SGDRegressor(max_iter=1000,
#                                  eta0=0.001,
#                                  learning_rate="constant",
#                                  random_state=42)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_val)
#
#         else:
#             raise ValueError("Invalid model type")
#
#         mse = mean_squared_error(y_val, y_pred)
#         r2  = r2_score(y_val, y_pred)
#
#         mse_list.append(mse)
#         r2_list.append(r2)
#
#         print(f"Fold {fold} | MSE: {mse:.4f} | R²: {r2:.4f}")
#
#     print("\nAverage MSE:", np.mean(mse_list))
#     print("Average R² :", np.mean(r2_list))
#     print("="*50)
#
#     return np.mean(mse_list), np.mean(r2_list)
#
#
# # =====================================================
# # MAIN
# # =====================================================
#
# if __name__ == "__main__":
#
#     data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
#     X = data.iloc[:, 0:5].values
#     y = data.iloc[:, 6].values
#
#     print("\n=== NORMAL EQUATION (SCRATCH) ===")
#     run_kfold(X, y, model_type="normal")
#
#     print("\n=== SGD (SCRATCH) ===")
#     run_kfold(X, y, model_type="sgd")
#
#     print("\n=== LINEAR REGRESSION (SKLEARN) ===")
#     run_kfold(X, y, model_type="sklearn_normal")
#
#     print("\n=== SGD REGRESSOR (SKLEARN) ===")
#     run_kfold(X, y, model_type="sklearn_sgd")

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# Load SONAR dataset
data = pd.read_csv("sonar.csv", header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1].map({'R':0, 'M':1})

model = LogisticRegression(max_iter=1000)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# ---------------------------
# 1. Without preprocessing
# ---------------------------
score1 = cross_val_score(model, X, y, cv=kfold)
print("Accuracy without preprocessing:", score1.mean())

# ---------------------------
# 2. Manual normalization
# ---------------------------
X_manual = (X - X.min()) / (X.max() - X.min())

score2 = cross_val_score(model, X_manual, y, cv=kfold)
print("Accuracy manual normalization:", score2.mean())

# ---------------------------
# 3. Sklearn normalization
# ---------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

score3 = cross_val_score(model, X_scaled, y, cv=kfold)
print("Accuracy sklearn normalization:", score3.mean())



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