import numpy as np

# Coefficient vector (theta)
theta = np.array([2, 3, 3])

# Data matrix X
X = np.array([
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
])

# Matrix multiplication
X_theta = X @ theta

print("Xθ =")
print(X_theta)
