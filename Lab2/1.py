import numpy as np

X = np.array([
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
])
X_mean = X - np.mean(X, axis=0)
cov_manual = (X_mean.T @ X_mean) / (X.shape[0] - 1)
print("Covariance matrix (manual):\n", cov_manual)
cov_numpy = np.cov(X, rowvar=False)
print("\nCovariance matrix (numpy):\n", cov_numpy)
