import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# =====================================================
# 1. Linear Regression FROM SCRATCH With Numpy
# =====================================================
class LinearRegressionFromScratch:

    def __init__(self, learning_rate=0.001, iterations=10000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.iterations):

            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # Gradients
            dw = (-2 / n_samples) * np.dot(X.T, (y - y_pred))
            db = (-2 / n_samples) * np.sum(y - y_pred)

            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 1000 == 0:
                loss = np.mean((y - y_pred) ** 2)
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def r2(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

# =====================================================
# 2. Linear Regression USING SCIKIT-LEARN
# =====================================================
class LinearRegressionSklearn:

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Store parameters explicitly
        self.weights = self.model.coef_
        self.bias = self.model.intercept_

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

# =====================================================
# 3. MAIN COMPARISON
# =====================================================
def main():

    # Load dataset
    df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

    X = df.iloc[:, 0:5].values
    y = df.iloc[:, 6].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=999
    )

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ============================
    # From Scratch Model
    # ============================
    scratch_model = LinearRegressionFromScratch(
        learning_rate=0.001,
        iterations=10000
    )
    scratch_model.fit(X_train_scaled, y_train)
    y_pred_scratch = scratch_model.predict(X_test_scaled)
    r2_scratch = scratch_model.r2(y_test, y_pred_scratch)

    # ============================
    # Scikit-Learn Model
    # ============================
    sklearn_model = LinearRegressionSklearn()
    sklearn_model.fit(X_train_scaled, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test_scaled)
    r2_sklearn = r2_score(y_test, y_pred_sklearn)

    # ============================
    # RESULTS
    # ============================
    print("\n========== MODEL COMPARISON ==========\n")

    print("FROM SCRATCH MODEL")
    print("Theta (Weights):", scratch_model.weights)
    print("Bias:", scratch_model.bias)
    print("R² Score:", r2_scratch)

    print("\nSCIKIT-LEARN MODEL")
    print("Theta (Weights):", sklearn_model.weights)
    print("Bias:", sklearn_model.bias)
    print("R² Score:", r2_sklearn)

    print("\n=====================================")


if __name__ == "__main__":
    main()
