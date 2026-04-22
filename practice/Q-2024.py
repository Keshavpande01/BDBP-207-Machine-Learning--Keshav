import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ISLP import load_data
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

# =====================================================
# CLASS DEFINITION
# =====================================================
class ML_Lab:

    def __init__(self):
        print("Initializing ML Lab...\n")

    # ================= Q1: SVM =================
    def svm_moons(self):
        print("Running SVM (Moons Dataset)...")

        X, y = make_moons(n_samples=200, noise=0.3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        # Linear SVM
        lin_svc = SVC(kernel='linear').fit(X_train, y_train)
        print("Linear Accuracy:", accuracy_score(y_test, lin_svc.predict(X_test)))

        # RBF SVM
        rbf_svc = SVC(kernel='rbf', gamma=1).fit(X_train, y_train)
        print("RBF Accuracy:", accuracy_score(y_test, rbf_svc.predict(X_test)))

        plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
        plt.title("Moons Dataset")
        plt.show()

    # ================= Q2: PCA + Hierarchical =================
    def nci60_pca_clustering(self):
        print("\nRunning PCA + Hierarchical Clustering (NCI60)...")

        data = load_data('NCI60')
        X = data['data']
        y = data['labels']

        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        y = y.to_numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        # Scatter
        plt.figure(figsize=(8,6))
        plt.scatter(X_pca[:,0], X_pca[:,1], c=pd.factorize(y)[0], cmap='tab10')
        plt.title("PCA (PC1 vs PC2)")
        plt.show()

        # Dendrogram
        Z = linkage(X_pca[:, :5], method='complete')
        plt.figure(figsize=(10,5))
        dendrogram(Z, labels=y, leaf_rotation=90)
        plt.title("Hierarchical Clustering")
        plt.show()

    # ================= Q3: KMeans From Scratch =================
    def kmeans_scratch(self):
        print("\nRunning KMeans from Scratch...")

        X = np.array([
            [1, 4], [1, 3], [0, 4],
            [5, 1], [6, 2], [4, 0]
        ])

        def initialize_centroids(X, K):
            indices = np.random.choice(X.shape[0], K, replace=False)
            return X[indices]

        def assign_clusters(X, centroids):
            distances = [np.linalg.norm(X - c, axis=1) for c in centroids]
            return np.argmin(distances, axis=0)

        def update_centroids(X, K, labels):
            new_centroids = []
            for i in range(K):
                pts = X[labels == i]
                if len(pts) == 0:
                    new_centroids.append(X[np.random.randint(0, X.shape[0])])
                else:
                    new_centroids.append(np.mean(pts, axis=0))
            return np.array(new_centroids)

        K = 2
        centroids = initialize_centroids(X, K)

        for _ in range(100):
            labels = assign_clusters(X, centroids)
            new_centroids = update_centroids(X, K, labels)

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
        plt.scatter(centroids[:,0], centroids[:,1], c='red', s=100)
        plt.title("KMeans (Scratch)")
        plt.show()


# =====================================================
# MAIN FUNCTION
# =====================================================
def main():
    lab = ML_Lab()

    lab.svm_moons()                # Q1
    lab.nci60_pca_clustering()    # Q2
    lab.kmeans_scratch()          # Q3


# =====================================================
if __name__ == "__main__":
    main()