import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Load data
# --------------------------------------------------
def load_data(filename):
    data = pd.read_csv(filename)
    X = data.iloc[:, 0:5].values.tolist()
    y = data.iloc[:, 5].values.tolist()
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=999)


# --------------------------------------------------
# Standardization (from scratch)
# --------------------------------------------------
def compute_mean_and_std(X):
    n_samples = len(X)
    n_features = len(X[0])

    mean = [0.0] * n_features
    std = [0.0] * n_features

    for j in range(n_features):
        mean[j] = sum(X[i][j] for i in range(n_samples)) / n_samples

    for j in range(n_features):
        std[j] = (
            sum((X[i][j] - mean[j]) ** 2 for i in range(n_samples)) / n_samples
        ) ** 0.5

    return mean, std


def normalize(X, mean, std):
    X_norm = []
    for i in range(len(X)):
        row = []
        for j in range(len(X[0])):
            row.append((X[i][j] - mean[j]) / std[j])
        X_norm.append(row)
    return X_norm


def add_bias(X):
    return [[1] + row for row in X]


def Matrix_Multiplication(X, y):
    n_sample = len(X)
    n_features = len(X[0])

    theta = n_sample * [0]

    XT = []
    for i in range(n_features):
        row = []
        for j in range(n_sample):
            row.append(X[j][i])
        XT.append(row)

    XTX = []
    for i in range(len(XT)):          # rows of XT
        row = []
        for j in range(len(X[0])):    # columns of X
            s = 0
            for k in range(len(X)):   # columns of XT / rows of X
                s += XT[i][k] * X[k][j]
            row.append(s)
        XTX.append(row)

def mat_vec_mul(X, theta):
    result = []
    for i in range(len(X)):
        s = 0
        for j in range(len(theta)):
            s += X[i][j] * theta[j]
        result.append(s)
    return result
def cost_function(X, y, theta):
    m = len(y)

    # Xθ
    y_hat = mat_vec_mul(X, theta)

    # (Xθ − y)^2
    cost = 0
    for i in range(m):
        error = y_hat[i] - y[i]
        cost += error ** 2

    # (1 / 2m) * sum
    return cost / (2 * m)

def matrix_inverse(A):
    A = np.array(A, dtype=float)
    return np.linalg.inv(A)

