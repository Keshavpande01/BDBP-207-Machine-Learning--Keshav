# -----------------------------
# Import required libraries
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Define the dataset
# -----------------------------
# Each row represents one observation: [X1, X2]
X = np.array([
    [1, 4],
    [1, 3],
    [0, 4],
    [5, 1],
    [6, 2],
    [4, 0]
])

k = 2  # Number of clusters
n = X.shape[0]  # Total number of observations

# -----------------------------
# (a) Plot original observations
# -----------------------------
plt.scatter(X[:, 0], X[:, 1], color='black')  # Plot all points
plt.title("Original Data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# -----------------------------
# (b) Randomly assign clusters
# -----------------------------
np.random.seed(42)  # Fix seed for reproducibility
labels = np.random.choice(k, n)  # Assign random cluster labels (0 or 1)

print("Initial Labels:", labels)


# -----------------------------
# Function to compute centroids
# -----------------------------
def compute_centroids(X, labels, k):
    centroids = []

    # Loop over each cluster
    for i in range(k):
        points = X[labels == i]  # Select points in cluster i

        # Compute centroid = mean of points
        centroid = points.mean(axis=0)
        centroids.append(centroid)

    return np.array(centroids)


# -----------------------------
# Function to assign clusters
# -----------------------------
def assign_clusters(X, centroids):
    new_labels = []

    # Loop over each data point
    for point in X:
        # Compute Euclidean distance to each centroid
        distances = [np.linalg.norm(point - c) for c in centroids]

        # Assign to nearest centroid
        new_labels.append(np.argmin(distances))

    return np.array(new_labels)


# -----------------------------
# (c), (d), (e) K-means iteration
# -----------------------------
while True:

    # Step (c): Compute centroids
    centroids = compute_centroids(X, labels, k)
    print("Centroids:\n", centroids)

    # Step (d): Assign clusters based on distance
    new_labels = assign_clusters(X, centroids)
    print("Updated Labels:", new_labels)

    # Step (e): Stop if labels do not change (convergence)
    if np.array_equal(labels, new_labels):
        print("Converged!")
        break

    # Update labels for next iteration
    labels = new_labels

# -----------------------------
# Final Output
# -----------------------------
print("\nFinal Cluster Labels:", labels)
print("Final Centroids:\n", centroids)

# -----------------------------
# (f) Plot final clustered data
# -----------------------------
colors = ['red', 'blue']  # Assign colors to clusters

for i in range(k):
    plt.scatter(
        X[labels == i][:, 0],  # X1 values of cluster i
        X[labels == i][:, 1],  # X2 values of cluster i
        color=colors[i],
        label=f'Cluster {i}'
    )

# Plot centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color='yellow',
    marker='X',
    s=200,
    label='Centroids'
)

plt.title("K-means Clustering Result")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()