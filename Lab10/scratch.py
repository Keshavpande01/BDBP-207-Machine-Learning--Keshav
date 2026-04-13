import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# -------------------- LOAD DATA --------------------
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------- GINI IMPURITY --------------------
def gini(y):
    classes = np.unique(y)
    impurity = 1
    for c in classes:
        p = np.sum(y == c) / len(y)
        impurity -= p**2
    return impurity

# -------------------- SPLIT FUNCTION --------------------
def split_data(X, y, feature, threshold):
    left = X[:, feature] <= threshold
    right = X[:, feature] > threshold
    return X[left], X[right], y[left], y[right]

# -------------------- BEST SPLIT --------------------
def best_split(X, y):
    best_score = float("inf")
    best_feature = None
    best_threshold = None

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])

        for t in thresholds:
            X_l, X_r, y_l, y_r = split_data(X, y, feature, t)

            if len(y_l) == 0 or len(y_r) == 0:
                continue

            # weighted gini
            w_left = len(y_l) / len(y)
            w_right = len(y_r) / len(y)

            score = w_left * gini(y_l) + w_right * gini(y_r)

            if score < best_score:
                best_score = score
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold

# -------------------- LEAF NODE --------------------
def create_leaf(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]

# -------------------- BUILD TREE --------------------
def build_tree(X, y, depth=0, max_depth=3):
    # stop conditions
    if len(np.unique(y)) == 1 or depth >= max_depth:
        return create_leaf(y)

    feature, threshold = best_split(X, y)

    if feature is None:
        return create_leaf(y)

    X_l, X_r, y_l, y_r = split_data(X, y, feature, threshold)

    return {
        "feature": feature,
        "threshold": threshold,
        "left": build_tree(X_l, y_l, depth+1, max_depth),
        "right": build_tree(X_r, y_r, depth+1, max_depth)
    }

# -------------------- PREDICTION --------------------
def predict_one(x, tree):
    if not isinstance(tree, dict):
        return tree

    if x[tree["feature"]] <= tree["threshold"]:
        return predict_one(x, tree["left"])
    else:
        return predict_one(x, tree["right"])

def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])

# -------------------- ACCURACY --------------------
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# -------------------- RUN --------------------
tree = build_tree(X_train, y_train, max_depth=3)

y_pred = predict(X_test, tree)

print("Predictions:", y_pred)
print("Accuracy:", accuracy(y_test, y_pred))
print(classification_report(y_test, y_pred))

