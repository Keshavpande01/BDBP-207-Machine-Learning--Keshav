import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score


# -----------------------------
# Load Data
# -----------------------------
def load_data():
    data = load_diabetes(as_frame=True)
    return data.data, data.target


# -----------------------------
# Tree Builder (Improved)
# -----------------------------
def build_tree(X, y, max_depth=5, min_samples=10, depth=0, max_features=None):
    n_samples = len(y)

    if (
        n_samples <= min_samples
        or depth >= max_depth
        or len(np.unique(y)) == 1
    ):
        return np.mean(y)

    features = X.columns

    # Feature subsampling
    if max_features:
        features = np.random.choice(features, max_features, replace=False)

    best_feature, best_threshold = None, None
    min_error = float("inf")

    for feature in features:
        values = np.sort(X[feature].values)
        thresholds = (values[:-1] + values[1:]) / 2  # midpoint splits

        for t in thresholds:
            left = X[feature] < t
            right = X[feature] >= t

            if left.sum() == 0 or right.sum() == 0:
                continue

            y_left = y[left]
            y_right = y[right]

            error = (
                np.sum((y_left - y_left.mean()) ** 2)
                + np.sum((y_right - y_right.mean()) ** 2)
            )

            if error < min_error:
                min_error = error
                best_feature = feature
                best_threshold = t

    if best_feature is None:
        return np.mean(y)

    left = X[best_feature] < best_threshold
    right = X[best_feature] >= best_threshold

    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left": build_tree(X[left], y[left], max_depth, min_samples, depth+1, max_features),
        "right": build_tree(X[right], y[right], max_depth, min_samples, depth+1, max_features),
    }


# -----------------------------
# Prediction
# -----------------------------
def predict_tree(tree, row):
    while isinstance(tree, dict):
        if row[tree["feature"]] < tree["threshold"]:
            tree = tree["left"]
        else:
            tree = tree["right"]
    return tree


def predict_forest(trees, X):
    preds = np.array([
        X.apply(lambda row: predict_tree(tree, row), axis=1)
        for tree in trees
    ])
    return preds.mean(axis=0)


# -----------------------------
# Bagging
# -----------------------------
def bagging(X_train, y_train, n_trees=10, max_depth=5, max_features=None):
    trees = []
    n = len(X_train)

    for _ in range(n_trees):
        # Bootstrap sampling
        indices = np.random.choice(n, n, replace=True)
        X_sample = X_train.iloc[indices]
        y_sample = y_train.iloc[indices]

        tree = build_tree(
            X_sample,
            y_sample,
            max_depth=max_depth,
            max_features=max_features
        )
        trees.append(tree)

    return trees


# -----------------------------
# Cross Validation
# -----------------------------
def kfold_evaluate(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        trees = bagging(X_train, y_train, n_trees=10, max_depth=5, max_features=5)
        y_pred = predict_forest(trees, X_val)

        scores.append(r2_score(y_val, y_pred))

    return np.mean(scores)


# -----------------------------
# Main
# -----------------------------
def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trees = bagging(X_train, y_train, n_trees=10, max_depth=5, max_features=5)

    y_pred = predict_forest(trees, X_test)

    print("\n--- Test Performance ---")
    print("R2 Score:", r2_score(y_test, y_pred))

    print("\n--- Cross Validation ---")
    cv_score = kfold_evaluate(X_train, y_train)
    print("CV R2:", cv_score)


if __name__ == "__main__":
    main()