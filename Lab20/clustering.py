import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from ISLP.cluster import compute_linkage

def generate_kmeans_clusters(X):
    kmeans2 = KMeans(n_clusters=2, random_state=2, n_init=20).fit(X)  # Apply K-Means with 2 clusters
    kmeans3 = KMeans(n_clusters=3, random_state=3, n_init=20).fit(X)  # Apply K-Means with 3 clusters
    return kmeans2, kmeans3  # Return both models

# Function to plot K-Means clusters with the given model and data
def plot_kmeans_clusters(X, kmeans_model, title):
    fig, ax = plt.subplots(figsize=(8, 8))   # Create a figure for the plot
    ax.scatter(X[:, 0], X[:, 1], c=kmeans_model.labels_)   # Scatter plot of the data points, colored by their assigned cluster
    ax.set_title(title)   # Set the title of the plot
    plt.show()  # Display the plot

# Function to perform hierarchical clustering with different linkage methods
def hierarchical_clustering(X, linkage_method='complete', scaled=False):
    scaler = StandardScaler()   # Create a scaler for standardizing the data
    # If scaling is required, standardize the data, otherwise use original data
    X_scaled = scaler.fit_transform(X) if scaled else X
    hc = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=linkage_method)  # Create the hierarchical clustering model
    hc.fit(X_scaled)   # Fit the model to the data
    linkage_matrix = compute_linkage(hc)   # Compute the linkage matrix for the dendrogram
    return hc, linkage_matrix   # Return the hierarchical clustering model and linkage matrix

# Function to plot the dendrogram for hierarchical clustering
def plot_dendrogram(linkage_matrix, title, color_threshold=None):
    fig, ax = plt.subplots(figsize=(8, 8))   # Create a figure for the plot
    cargs = {'above_threshold_color': 'black'}   # Set default color for dendrogram branches
    if color_threshold is not None:
        cargs['color_threshold'] = color_threshold   # Set color threshold if provided
    dendrogram(linkage_matrix, ax=ax, **cargs)   # Create the dendrogram plot
    ax.set_title(title)   # Set the title of the plot
    if color_threshold:
        ax.axhline(color_threshold, c='r', linewidth=4)   # Draw a red horizontal line at the color threshold
    plt.show()

def main():
    # Generate synthetic data for clustering
    np.random.seed(0)  # Set random seed for reproducibility
    X = np.random.standard_normal((50, 2))  # Generate random data
    X[:25, 0] += 3  # Modify some data points to create clusters
    X[:25, 1] -= 4  # Modify data points

    # Generate K-Means clusters and plot the results
    kmeans2, kmeans3 = generate_kmeans_clusters(X)
    plot_kmeans_clusters(X, kmeans2, "K-Means Clustering with K=2")  # Plot clusters for K=2
    plot_kmeans_clusters(X, kmeans3, "K-Means Clustering with K=3")  # Plot clusters for K=3

    # Perform hierarchical clustering and plot the dendrogram
    _, linkage_matrix = hierarchical_clustering(X, linkage_method='complete')
    plot_dendrogram(linkage_matrix, "Hierarchical Clustering with Complete Linkage")  # Plot dendrogram for unscaled data

    # Perform hierarchical clustering on scaled data and plot the dendrogram
    _, linkage_scaled = hierarchical_clustering(X, linkage_method='complete', scaled=True)
    plot_dendrogram(linkage_scaled, "Hierarchical Clustering with Scaled Features")  # Plot dendrogram for scaled data

if __name__ == "__main__":
    main()