import numpy as np
import pandas as pd


# ==============================
# LOAD DATA
# ==============================
def load_data(filename):
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y = np.where(y == 'M', 1, 0)
    return X, y


# ==============================
# TRAIN TEST SPLIT
# ==============================
def train_test_split(X, y, test_size=0.4):
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split = int(len(X) * (1 - test_size))

    train_idx = indices[:split]
    test_idx = indices[split:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# ==============================
# K-FOLD
# ==============================
def K_fold(X, y, n_folds=10):
    indices = np.arange(len(X))  # if 5 samples = [0, 1, 2, 3, 4]
    np.random.shuffle(indices)   # randomizes = [2, 0, 4, 1, 3]

    folds = np.array_split(indices, n_folds)
    # Fold1 → [2, 0]
    # Fold2 → [4, 1]
    # Fold3 → [3]

    for i in range(n_folds): # Each iteration = one validation round
        val_idx = folds[i]  # select validation set
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        # “Take all folds except the current one and combine them into training data”

        yield train_idx, val_idx

# Model
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(X, y, lr=0.01, epochs=1000): # Changed n_folds to epochs
    w = np.zeros(X.shape[1])
    b = 0

    for _ in range(epochs):
        z = X @ w + b
        y_pred = sigmoid(z)

        # Gradient calculation (Standard: pred - actual)
        dw = (X.T @ (y_pred - y)) / len(y)
        db = np.mean(y_pred - y)

        # Update parameters (Subtract because we move against the gradient)
        w -= lr * dw
        b -= lr * db

    return w, b

def predict(X, w, b):
    #convert probabilites _> class label(0,1)
    prob = sigmoid(X @ w + b)
    return (prob >= 0.5).astype(int)

def accuracy(y, y_pred):
    return np.mean((y == y_pred))



# MAIN

if __name__ == "__main__":

    # Load data
    X, y = load_data("sonar.csv")

    # Apply K-Fold on FULL DATA (or you can use X_train)
    # Updated main loop
    for i, (train_idx, val_idx) in enumerate(K_fold(X, y, 10), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Now you can train and validate to see how it performs across folds
        w_fold, b_fold = train(X_tr, y_tr, lr=0.1, epochs=500)
        preds = predict(X_val, w_fold, b_fold)
        print(f"Fold {i:<2} | Acc: {accuracy(y_val, preds):.4f}")
    # Split
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    print("After Train-Test Split:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)
    print("-" * 40)

    # Train model
    w, b = train(X_train, y_train)
    print("Model Parameters:")
    print("Weights shape:", w.shape)
    print("Bias:", b)
    print("-" * 40)

    # Predict
    y_pred = predict(X_test, w, b)
    print("Predictions:")
    print("First 10 predictions:", y_pred[:10])
    print("First 10 actual     :", y_test[:10])
    print("-" * 40)

    # Accuracy
    acc = accuracy(y_test, y_pred)
    print("Final Result:")
    print("Test Accuracy:", acc)



