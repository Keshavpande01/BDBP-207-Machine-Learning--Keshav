import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ISLP import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------ LOAD DATA ------------------
data = load_data("Khan")
X = pd.DataFrame(data['xtrain'])
y = pd.Series(data['ytrain'], name="Cancer_Type")

print("Shape of dataset:", X.shape)
print("Unique classes:", np.unique(y))

# ------------------ BASIC EDA ------------------
print("\nSummary Statistics:")
print(X.describe().iloc[:, :5])  # show first 5 features

print("\nMissing Values:", X.isnull().sum().sum())

# ------------------ STANDARDIZATION ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------ PCA ------------------
pca = PCA()
X_pca_full = pca.fit_transform(X_scaled)

# Explained variance (Scree Plot)
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Scree Plot (Khan Dataset)")
plt.grid()
plt.show()

# Reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance (2 PCs):", pca.explained_variance_ratio_)

# ------------------ PCA VISUALIZATION ------------------
plt.figure(figsize=(8,6))

for label in np.unique(y):
    plt.scatter(
        X_pca[y == label, 0],
        X_pca[y == label, 1],
        label=f"Class {label}",
        alpha=0.7
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization (Khan Dataset)")
plt.legend(title="Cancer Type")
plt.grid()
plt.show()

# ------------------ K-MEANS (ELBOW METHOD) ------------------
wcss = []
K_range = range(1, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, wcss, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid()
plt.show()

# ------------------ FINAL KMEANS ------------------
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# ------------------ CLUSTER VISUALIZATION ------------------
plt.figure(figsize=(8,6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap='viridis',
    alpha=0.7
)

# Centroids
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c='red',
    marker='X',
    s=200,
    label='Centroids'
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering on PCA Data")
plt.legend()
plt.grid()
plt.show()

# ------------------ CLUSTER QUALITY ------------------
sil_score = silhouette_score(X_pca, clusters)
print("\nSilhouette Score:", sil_score)

# ------------------ COMPARISON ------------------
comparison = pd.DataFrame({
    "True_Label": y,
    "Cluster": clusters
})

print("\nSample Comparison:")
print(comparison.head(10))