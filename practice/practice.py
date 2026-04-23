# ============================================
# NCI60 PCA + Visualization + Clustering
# ============================================

from ISLP import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

# -------------------------------
# 1. Load Data
# -------------------------------
NCI60 = load_data('NCI60')

X = NCI60['data']      # gene expression data
y = NCI60['labels']    # cancer types

# Convert labels to 1D array (IMPORTANT for dendrogram)
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]
y = y.to_numpy()

# -------------------------------
# 2. Standardize Data
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 3. Perform PCA
# -------------------------------
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Create DataFrame for plotting
pca_df = pd.DataFrame(X_pca[:, :3], columns=['PC1', 'PC2', 'PC3'])
pca_df['CancerType'] = y

# -------------------------------
# 4(a). Scatter Plots
# -------------------------------

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df , x='PC1', y='PC2',hue='CancerType', palette='tab10')
plt.title('PC1 vs PC2')
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df,x='PC1',y='PC3',hue='CancerType',palette='tab10')
plt.title('PC1 vs PC3')
plt.legend(bbox_to_anchor=(1.05,1),loc='upper left')
plt.tight_layout()
plt.show()

# -------------------------------
# 4(b). Variance Explained
# -------------------------------
explained_var = pca.explained_variance_ratio_
cummulative_var = np.cumsum(explained_var)

plt.figure(figsize=(8, 6))
plt.plot(explained_var)
plt.plot(cummulative_var,marker='o',label='Individual Variance')
plt.plot(cummulative_var,marker='s',label='Cumulative Variance')
plt.xlabel('Principal component')
plt.ylabel('Variance Explained')
plt.title('Variance Explained by Principal Components')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# 4(c). Hierarchical Clustering
# -------------------------------
# Use first 5 principal components
X_pca_subset = X_pca[:, :5]
Z = linkage(X_pca_subset, method='complete')
X_pca_subset = X_pca[:, :5]

# Perform clustering (complete linkage)
Z = linkage(X_pca_subset, method='complete')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z, labels=y, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram (Complete Linkage)')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()


