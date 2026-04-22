import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to binary classification (class 0 vs others)
y = np.where(y == 0, -1, 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# Decision Stump (Weak Learner)
# -----------------------------
class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[X[:, self.feature_index] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_index] < self.threshold] = 1

        return predictions

# -----------------------------
# AdaBoost Classifier
# -----------------------------
class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights
        w = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float("inf")

            # Try all features and thresholds
            for feature in range(n_features):
                thresholds = np.unique(X[:, feature])

                for threshold in thresholds:
                    for polarity in [1, -1]:
                        predictions = np.ones(n_samples)
                        predictions[X[:, feature] < threshold] = -1
                        if polarity == -1:
                            predictions = -predictions

                        # Compute weighted error
                        error = np.sum(w[y != predictions])

                        if error < min_error:
                            min_error = error
                            stump.feature_index = feature
                            stump.threshold = threshold
                            stump.polarity = polarity

            # Compute alpha (model weight)
            eps = 1e-10  # avoid division by zero
            alpha = 0.5 * np.log((1 - min_error + eps) / (min_error + eps))

            # Update weights
            predictions = stump.predict(X)
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)

            # Save model
            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        model_preds = []

        for alpha, model in zip(self.alphas, self.models):
            preds = model.predict(X)
            model_preds.append(alpha * preds)

        y_pred = np.sign(np.sum(model_preds, axis=0))
        return y_pred

# -----------------------------
# Train AdaBoost
# -----------------------------
model = AdaBoost(n_estimators=10)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"AdaBoost Accuracy: {accuracy:.2f}")