import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

class DecisionTreeID3:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    # Entropy
    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-9))  # avoid log(0)

    # Split dataset
    def split(self, X, y, feature, threshold):
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        return X[left_idx], X[right_idx], y[left_idx], y[right_idx]

    # Information Gain
    def information_gain(self, y, y_l, y_r):
        parent_entropy = self.entropy(y)
        n = len(y)

        if len(y_l) == 0 or len(y_r) == 0:
            return 0

        weighted_entropy = (len(y_l)/n)*self.entropy(y_l) + \
                           (len(y_r)/n)*self.entropy(y_r)

        return parent_entropy - weighted_entropy

    # Best split
    def best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                X_l, X_r, y_l, y_r = self.split(X, y, feature, threshold)

                gain = self.information_gain(y, y_l, y_r)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # Build tree
    def build_tree(self, X, y, depth):
        if len(y) == 0:
            return None

        if len(np.unique(y)) == 1 or depth >= self.max_depth:
            return np.bincount(y).argmax()

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return np.bincount(y).argmax()

        X_l, X_r, y_l, y_r = self.split(X, y, feature, threshold)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self.build_tree(X_l, y_l, depth + 1),
            "right": self.build_tree(X_r, y_r, depth + 1)
        }

    #  Fit
    def fit(self, X, y):
        self.tree = self.build_tree(X, y, 0)

    #  Predict one
    def predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree

        if x[tree["feature"]] <= tree["threshold"]:
            return self.predict_one(x, tree["left"])
        else:
            return self.predict_one(x, tree["right"])

    # Predict
    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])


# Train
model = DecisionTreeID3(max_depth=3)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)