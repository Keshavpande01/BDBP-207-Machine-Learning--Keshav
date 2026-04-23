# ============================================
# USArrests (Manual Data) + PCA + Clustering + PRINTS
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram

# -------------------------------
# 1. MANUAL DATA
# -------------------------------
data = {
    "State": ["Alabama","Alaska","Arizona","Arkansas","California","Colorado"],
    "Murder": [13.2,10.0,8.1,8.8,9.0,7.9],
    "Assault": [236,263,294,190,276,204],
    "UrbanPop": [58,48,80,50,91,78],
    "Rape": [21.2,44.5,31.0,19.5,40.6,38.7]
}

df = pd.DataFrame(data)
df.set_index("State", inplace=True)

# -------------------------------
# 🔥 PRINT ORIGINAL DATA
# -------------------------------
print("\n===== ORIGINAL DATA =====")
print(df)

# -------------------------------
# 2. Standardize
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

print("\n===== STANDARDIZED DATA =====")
print(np.round(X_scaled, 2))

# -------------------------------
# 3. PCA
# -------------------------------
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
pca_df['State'] = df.index

print("\n===== PCA VALUES (PC1, PC2) =====")
print(pca_df)

# -------------------------------
# 4. PCA Plot
# -------------------------------
plt.scatter(pca_df['PC1'], pca_df['PC2'])

for i, state in enumerate(pca_df['State']):
    plt.annotate(state, (pca_df['PC1'][i], pca_df['PC2'][i]))

plt.title("PCA - USArrests")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# -------------------------------
# 5. K-means
# -------------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

pca_df['Cluster'] = clusters

# 🔥 PRINT LABELS
print("\n===== K-MEANS CLUSTER LABELS =====")
for state, label in zip(df.index, clusters):
    print(f"{state} -> Cluster {label}")

# 🔥 PRINT CENTROIDS
print("\n===== CENTROIDS (STANDARDIZED SPACE) =====")
print(np.round(kmeans.cluster_centers_, 2))

# Plot clusters
plt.figure()
for i in range(2):
    subset = pca_df[pca_df['Cluster'] == i]
    plt.scatter(subset['PC1'], subset['PC2'], label=f'Cluster {i}')

plt.legend()
plt.title("K-means Clustering")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# -------------------------------
# 6. Hierarchical Clustering
# -------------------------------
Z = linkage(X_scaled, method='complete')

plt.figure()
dendrogram(Z, labels=df.index)
plt.title("Hierarchical Clustering")
plt.show()