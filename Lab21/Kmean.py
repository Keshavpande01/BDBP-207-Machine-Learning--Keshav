### Implement K-Means algorithm ground-up using Python.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

iris = load_iris()   # load the iris dataset
X_df = pd.DataFrame(iris.data, columns=iris.feature_names)   # Since, it is an unsupervised learning, so only features (X) is given and not the target variable (y).
X = X_df.values   # convert to NumPy array for processing

# EDA
print("Overview of the dataset: ")
print(X_df.info() )   # Overview of the dataset (no categorical values present)
print("Statistical summary of dataset: ")
print(X_df.describe())   # Statistical summary of the dataset
print("Unique values of dataset: ")
print(X_df.nunique())    # displays the unique values
print("Column names of dataset: ")
print(X_df.columns)    # gives feature column names
print("Duplicate values of dataset: ")
print(X_df.duplicated().sum())   # Only 1 duplicate value is there which can be ignored.
print("Null values in the dataset: ")
print(X_df.isnull().sum())   # No null values present in the data

def cluster_assignments(X, k):
    np.random.seed(42)   # ensures reproducibility
    return np.random.randint(1, k+1, size=X.shape[0])   # assigns each of the n data points a random cluster label from 1 to k.

def compute_centroid(X, labels, k):
    centroids = []   # for storing the centroids values
    for i in range(1, k+1):   # it will iterate through each cluster
        cluster_points = X[labels==i]   # it will consider all the data points of each cluster
        if len(cluster_points) > 0:   # if a cluster has data points in it, then it will compute the centroid, i.e., average or mean of all the data points present in that cluster.
            centroid = cluster_points.mean(axis=0)   # axis=0: column-wise mean
        else:
            centroid = X[np.random.randint(0, X.shape[0])]   # else, randomly assign a centroid from the dataset.
        centroids.append(centroid)
    return np.array(centroids)

def compute_euclidean_distance(X, centroids):   # Formula: sqrt((x2-x1)^2 + (y2-y1)^2)
    distances = np.linalg.norm(X[:,np.newaxis]-centroids, axis=2)    # X[:, np.newaxis]-centroids: creates a distance matrix ; axis=2: collapses the innermost dimension (the feature space).
    return np.argmin(distances, axis=1)+1   # finds the index of the nearest centroid (min. distance); +1: bec. we want labels starting from 1, not 0.

def update_labels(X, k, max_iters=100):
    labels = cluster_assignments(X, k)   # Start with a random cluster label (1 to k) for each observation.
    for _ in range(max_iters):   # At each iteration:
        centroids = compute_centroid(X, labels , k)   # compute the centroids of the current clusters.
        new_labels = compute_euclidean_distance(X, centroids)  # reassign each point to the closest centroid.
        if np.array_equal(labels, new_labels):  # If the labels don’t change, then convergence has happened — so stop early.
            break
        labels = new_labels   # Otherwise, repeat with updated labels.
    return labels, centroids   # Finally, return the cluster assignments and the final centroids.

def main():
    # Visualize actual iris classes using PCA (Optional: just for comparison purpose)
    pca = PCA(n_components=2)   # create a PCA object to reduce the dimensions from 4D to 2D
    X_2d = pca.fit_transform(X)   # fit_transform() - learns and applies PCA
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=iris.target, cmap='viridis', s=50)
    plt.title("Actual Iris Classes")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

    # K-means clustering -
    k = 2   # try different values of k; we'll find that k=2 gives max. silhouette score, i.e., closer to 1.0, that's why we chose k as 2 (in unsupervised learning, we don't know 'y')
    # In case of k=2, we get only setosa and versicolor-virginica mix.
    # If we choose k as 3, then we will get exact clusters/classes as it is there in the dataset (setosa, versicolor, virginica).
    labels, centroids = update_labels(X, k)
    score = silhouette_score(X, labels-1)   # labels adjusted to 0-based for silhouette
    print(f"Silhouette score for k={k} is {score:.3f}")

    # Reuse PCA for plotting K-Means result -
    centroids_2d = pca.transform(centroids)   # projects the high-dimensional centroids into 2D PCA space.
    # X_2d[:, 0] & Z_2d[:, 1]: selects the 1st & 2nd PC of the 2D PCA-transformed data, c=labels: sets the color of each point based on its cluster assignment,
    # cmap='viridis': uniform colour map ranging from purple to yellow, s=50: size of each scatter dot (default: 20)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='X', s=200)    # red 'X' represents the centroid of its cluster
    plt.title("K-Means Clustering (Iris dataset)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

if __name__ == "__main__":
    main()