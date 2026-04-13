# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
#
#
# # ── 1. Load Data ──────────────────────────────────────────────────────────────
# def load_data(filename):
#     col_names = [f"F{i}" for i in range(1, 61)] + ["Label"]
#     df = pd.read_csv(filename, header=None, names=col_names)
#     return df
#
# # ── 2. Define X and y ─────────────────────────────────────────────────────────
# def define_X_y(df):
#     X = df.iloc[:, :-1].values.astype(float)
#     y = df.iloc[:, -1].values
#     return X, y
#
# # ── 3. Split Data ─────────────────────────────────────────────────────────────
# def split_data(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42)
#     return X_train, X_test, y_train, y_test
#
#
# # ── 4. Train Model ────────────────────────────────────────────────────────────
# def train_model(X_train, y_train):
#     model = LogisticRegression(max_iter=2000, random_state=42)
#     model.fit(X_train, y_train)
#     return model
#
# # ── 5. Predict ────────────────────────────────────────────────────────────────
# def get_predictions(model, X_test):
#     y_pred = model.predict(X_test)
#     return y_pred
#
# # ── 6. Accuracy ───────────────────────────────────────────────────────────────
# def get_accuracy(y_test, y_pred):
#     accuracy = accuracy_score(y_test, y_pred)
#     return accuracy
#
#
# # ── Main ──────────────────────────────────────────────────────────────────────
# def main():
#     # 1. Load
#     df = load_data("sonar.csv")
#     print(df.head())
#     print(df.shape)
#     print(df.columns)
#
#     # 2. Define X and y
#     X, y = define_X_y(df)
#
#     # 3. Split
#     X_train, X_test, y_train, y_test = split_data(X, y)
#
#     # ── Without Preprocessing ─────────────────────────────────────────────────
#     model_raw     = train_model(X_train, y_train)
#     y_pred_raw    = get_predictions(model_raw, X_test)
#     acc_raw       = get_accuracy(y_test, y_pred_raw)
#
#     # ── Manual MinMax Normalization ───────────────────────────────────────────
#     x_min         = X_train.min(axis=0)
#     x_max         = X_train.max(axis=0)
#     r             = np.where(x_max - x_min == 0, 1, x_max - x_min)
#     X_train_manual = (X_train - x_min) / r
#     X_test_manual  = (X_test  - x_min) / r
#
#     model_manual  = train_model(X_train_manual, y_train)
#     y_pred_manual = get_predictions(model_manual, X_test_manual)
#     acc_manual    = get_accuracy(y_test, y_pred_manual)
#
#     # ── sklearn StandardScaler ────────────────────────────────────────────────
#     scaler         = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled  = scaler.transform(X_test)
#
#     model_scaled   = train_model(X_train_scaled, y_train)
#     y_pred_scaled  = get_predictions(model_scaled, X_test_scaled)
#     acc_scaled     = get_accuracy(y_test, y_pred_scaled)
#
#     # ── Results ───────────────────────────────────────────────────────────────
#     print(f"\n{'Method':<25} {'Accuracy':>10} {'Accuracy (%)':>14}")
#     print("-" * 52)
#     print(f"{'No Preprocessing':<25} {acc_raw:>10.4f} {acc_raw*100:>13.2f}%")
#     print(f"{'Manual MinMax':<25} {acc_manual:>10.4f} {acc_manual*100:>13.2f}%")
#     print(f"{'sklearn StandardScaler':<25} {acc_scaled:>10.4f} {acc_scaled*100:>13.2f}%")
#
# if __name__ == '__main__':
#     main()

#------------------------Logistic Regression From Scratch -------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# ───────────────── Logistic Regression From Scratch ─────────────────
class LogisticRegressionScratch:

    def __init__(self, lr=0.01, n_iters=2000):
        self.lr = lr
        self.n_iters = n_iters

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):

            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):

        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)

        return np.where(y_pred >= 0.5, 1, 0)


# ───────────────── Load Dataset ─────────────────
def load_data(filename):

    col_names = [f"F{i}" for i in range(1, 61)] + ["Label"]
    df = pd.read_csv(filename, header=None, names=col_names)

    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values

    # Convert R/M → 0/1
    y = np.where(y == 'R', 0, 1)

    return X, y


# ───────────────── Main Program ─────────────────
def main():

    # Load dataset
    X, y = load_data("sonar.csv")

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ── 1. Without preprocessing ──
    model_raw = LogisticRegressionScratch()
    model_raw.fit(X_train, y_train)
    y_pred_raw = model_raw.predict(X_test)
    acc_raw = accuracy_score(y_test, y_pred_raw)

    # ── 2. Manual MinMax normalization ──────────────────
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)

    r = np.where(x_max - x_min == 0, 1, x_max - x_min)

    X_train_norm = (X_train - x_min) / r
    X_test_norm = (X_test - x_min) / r

    model_norm = LogisticRegressionScratch()
    model_norm.fit(X_train_norm, y_train)
    y_pred_norm = model_norm.predict(X_test_norm)
    acc_norm = accuracy_score(y_test, y_pred_norm)

    # ── 3. StandardScaler ────────────────────────
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_scaled = LogisticRegressionScratch()
    model_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = model_scaled.predict(X_test_scaled)
    acc_scaled = accuracy_score(y_test, y_pred_scaled)

    # ── Results ────────────────────────────────
    print("\nDataset shape:", X.shape)

    print(f"\n{'Method':<25} {'Accuracy':>10} {'Accuracy (%)':>14}")
    print("-"*52)

    print(f"{'No Preprocessing':<25} {acc_raw:>10.4f} {acc_raw*100:>13.2f}%")
    print(f"{'Manual MinMax':<25} {acc_norm:>10.4f} {acc_norm*100:>13.2f}%")
    print(f"{'StandardScaler':<25} {acc_scaled:>10.4f} {acc_scaled*100:>13.2f}%")


if __name__ == "__main__":
    main()