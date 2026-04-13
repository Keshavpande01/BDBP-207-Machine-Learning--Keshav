import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# LOAD DATA
df = pd.read_csv("sonar.csv", header=None)
X = df.iloc[:, :-1].values
y = (df.iloc[:, -1] == 'M').astype(int).values


# STEP 1: TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)
print("-" * 40)


# STEP 2: K-FOLD ON TRAINING
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = []
for i, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # Scale (NO LEAKAGE)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    # Train
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)

    # Validate
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    scores.append(acc)

    print(f"Fold {i:<2} | Accuracy: {acc:.4f}")

print("\nMean CV Accuracy:", np.mean(scores))
print("Std Dev:", np.std(scores))
print("-" * 40)


# STEP 3: FINAL TRAIN ON FULL TRAIN SET
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


# STEP 4: FINAL TEST EVALUATION
y_test_pred = model.predict(X_test_scaled)

print("Final Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["Rock", "Mine"]))