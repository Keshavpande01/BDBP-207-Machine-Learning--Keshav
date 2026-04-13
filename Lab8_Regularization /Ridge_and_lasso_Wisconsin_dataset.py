import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ───────── Load Dataset ─────────
data = load_breast_cancer()
X = data.data
y = data.target

# ───────── Train-Test Split ─────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ───────── Feature Scaling ─────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 🔵 RIDGE CLASSIFIER (L2 Regularization)
# ============================================================

ridge_model = RidgeClassifier(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

ridge_pred = ridge_model.predict(X_test_scaled)

print("\n===== RIDGE CLASSIFIER =====")
print("Accuracy:", accuracy_score(y_test, ridge_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, ridge_pred))
print("\nClassification Report:\n", classification_report(y_test, ridge_pred))


# ============================================================
# LASSO CLASSIFIER (L1 Regularization)
# ============================================================

lasso_model = LogisticRegression(
    penalty='l1',
    solver='liblinear',   # required for L1
    C=1.0
)

lasso_model.fit(X_train_scaled, y_train)

lasso_pred = lasso_model.predict(X_test_scaled)

print("\n===== LASSO CLASSIFIER =====")
print("Accuracy:", accuracy_score(y_test, lasso_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, lasso_pred))
print("\nClassification Report:\n", classification_report(y_test, lasso_pred))


# 🔍 Feature Selection Insight (Lasso)

print("\nNon-zero coefficients (Lasso):")
print(np.sum(lasso_model.coef_ != 0))
