import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# -----------------------------
# Load Dataset
# -----------------------------
def load_data():
    data = load_diabetes()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------
# Metrics
# -----------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)


# -----------------------------
# Decision Tree Regressor (CART)
# -----------------------------
class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=10, min_gain=1e-3):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.tree = None
        self.feature_importance = None

    # 🔹 SSE
    def sse(self, y):
        if len(y) == 0:
            return 0
        return np.sum((y - np.mean(y)) ** 2)

    # 🔹 Find best split
    def best_split(self, X, y):
        best_feature, best_threshold = None, None
        min_error = float("inf")

        n_samples, n_features = X.shape

        for j in range(n_features):
            values = np.sort(X[:, j])
            thresholds = (values[:-1] + values[1:]) / 2

            for t in thresholds:
                left_mask = X[:, j] < t
                right_mask = X[:, j] >= t

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                error = self.sse(y_left) + self.sse(y_right)

                if error < min_error:
                    min_error = error
                    best_feature = j
                    best_threshold = t

        return best_feature, best_threshold, min_error

    # 🔹 Build tree
    def build_tree(self, X, y, depth=0):
        n_samples = len(y)

        # Stop conditions
        if (
            n_samples < self.min_samples_split
            or depth >= self.max_depth
            or len(np.unique(y)) == 1
        ):
            leaf_value = np.mean(y)
            print("  " * depth + f"Leaf → {leaf_value:.2f}")
            return leaf_value

        feature, threshold, error = self.best_split(X, y)

        if feature is None:
            leaf_value = np.mean(y)
            print("  " * depth + f"Leaf → {leaf_value:.2f}")
            return leaf_value

        # Compute gain
        parent_error = self.sse(y)

        left_mask = X[:, feature] < threshold
        right_mask = X[:, feature] >= threshold

        y_left = y[left_mask]
        y_right = y[right_mask]

        child_error = self.sse(y_left) + self.sse(y_right)
        gain = parent_error - child_error

        # Store feature importance
        self.feature_importance[feature] += gain

        # Pruning
        if gain < self.min_gain:
            leaf_value = np.mean(y)
            print("  " * depth + f"Pruned → {leaf_value:.2f}")
            return leaf_value

        print("  " * depth + f"[Depth {depth}] Split → X{feature} < {threshold:.4f} | Gain {gain:.2f}")

        # Recursive build
        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree
        }

    # 🔹 Fit
    def fit(self, X, y):
        self.feature_importance = np.zeros(X.shape[1])
        self.tree = self.build_tree(X, y)

    # 🔹 Predict one
    def predict_one(self, x, tree):
        while isinstance(tree, dict):
            if x[tree["feature"]] < tree["threshold"]:
                tree = tree["left"]
            else:
                tree = tree["right"]
        return tree

    # 🔹 Predict
    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])

    # 🔹 Print tree
    def print_tree(self, tree=None, depth=0):
        if tree is None:
            tree = self.tree

        if not isinstance(tree, dict):
            print("  " * depth + f"→ {tree:.2f}")
            return

        print("  " * depth + f"[X{tree['feature']} < {tree['threshold']:.4f}]")

        print("  " * depth + "Left:")
        self.print_tree(tree["left"], depth + 1)

        print("  " * depth + "Right:")
        self.print_tree(tree["right"], depth + 1)

    # 🔹 Feature importance
    def get_feature_importance(self):
        total = np.sum(self.feature_importance)
        if total == 0:
            return self.feature_importance
        return self.feature_importance / total


# -----------------------------
# Main
# -----------------------------
def main():
    X_train, X_test, y_train, y_test = load_data()

    model = DecisionTreeRegressor(max_depth=5, min_samples_split=10)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n--- Evaluation ---")
    print("MSE:", mse(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    print("\n--- Tree Structure ---")
    model.print_tree()

    print("\n--- Feature Importance ---")
    importance = model.get_feature_importance()
    for i, val in enumerate(importance):
        print(f"Feature {i}: {val:.4f}")

    # Compare with sklearn
    from sklearn.tree import DecisionTreeRegressor as SkTree

    sk_model = SkTree(max_depth=5)
    sk_model.fit(X_train, y_train)

    sk_pred = sk_model.predict(X_test)

    print("\n--- Sklearn Comparison ---")
    print("Sklearn MSE:", mse(y_test, sk_pred))
    print("Sklearn R2:", r2_score(y_test, sk_pred))


if __name__ == "__main__":
    main()