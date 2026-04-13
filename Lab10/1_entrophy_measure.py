import numpy as np


def entropy(y):


    # Get unique classes and their counts
    classes, counts = np.unique(y, return_counts=True)

    # Convert counts to probabilities
    probabilities = counts / len(y)

    # Compute entropy
    entropy_value = 0
    for p in probabilities:
        if p > 0:
            entropy_value -= p * np.log2(p)

    return entropy_value

# Example dataset
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

print("Entropy:", entropy(y))