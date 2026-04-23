# ============================================
# PRINCIPAL COMPONENT ANALYSIS (FINAL CLEAN)
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# ============================================
# PCA ON USArrests
# ============================================
def PCA_custom():

    data = {
        "State": ["Alabama","Alaska","Arizona","Arkansas","California",
                  "Colorado","Connecticut","Delaware","Florida","Georgia"],
        "Murder": [13.2,10.0,8.1,8.8,9.0,7.9,3.3,5.9,15.4,17.4],
        "Assault": [236,263,294,190,276,204,110,238,335,211],
        "UrbanPop": [58,48,80,50,91,78,77,72,80,60],
        "Rape": [21.2,44.5,31.0,19.5,40.6,38.7,11.1,15.8,31.9,25.8]
    }

    df = pd.DataFrame(data).set_index("State")

    print("\n===== ORIGINAL DATA =====")
    print(df)

    # Standardize
    X_scaled = StandardScaler().fit_transform(df)

    # PCA
    pca = PCA()
    scores = pca.fit_transform(X_scaled)

    # ================== INTERPRETATION ==================
    print("\n" + "="*50)
    print("        PCA INTERPRETATION")
    print("="*50)

    explained_var = pca.explained_variance_ratio_

    print("\nVariance Explained:")
    for i, var in enumerate(explained_var):
        print(f"  PC{i+1}: {var:.4f}")

    print(f"\nTotal (PC1 + PC2): {sum(explained_var[:2]):.4f}")

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(pca.components_))],
        index=df.columns
    )

    print("\nLoadings:\n", loadings)

    print("\nKey Contributors:")
    print("PC1:", loadings["PC1"].abs().sort_values(ascending=False).head(2).index.tolist())
    print("PC2:", loadings["PC2"].abs().sort_values(ascending=False).head(2).index.tolist())

    print("\nConclusion:")
    print("PC1 → overall crime level")
    print("PC2 → urban population variation")

    print("="*50)


# ============================================
# PCA + SVM (SONAR DATASET)
# ============================================
def PCA_OnDataSet():

    data = pd.read_csv("sonar.csv", header=None)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    y = LabelEncoder().fit_transform(y)
    X = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X)

    print("\n===== PCA DIMENSION REDUCTION =====")
    print("Original features:", X.shape[1])
    print("Reduced features:", X_pca.shape[1])

    # K-Fold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    acc_list = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_pca, y), 1):

        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = SVC()
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        acc_list.append(acc)

        print(f"Fold {fold}: Accuracy = {acc:.4f}")

    print("\nAverage Accuracy:", np.mean(acc_list))


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    PCA_custom()
    PCA_OnDataSet()