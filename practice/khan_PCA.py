import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1,4],
    [2,5],
    [3,6],
    [4,7],
    [5,8],
    [6,9],
    [7,10],
])

k = 2  # Number of clusters
n = X.shape[0]  # Total number of observation

plt.scatter(X[:, 0], X[:, 1], c='red', marker='o', s=100)
plt.title("Original Data")

# Randomly assign clusters
np.random.seed(42)       # Fix seed for reproducibility
labels = np.random.choice # Assign random clusters label (0 or 1 )
print(labels)

def compute_centroid(X ,labels , k):
    