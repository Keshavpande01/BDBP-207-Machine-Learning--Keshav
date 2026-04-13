import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("sonar.csv", header=None)
X = df.iloc[:, :-1].values
y = (df.iloc[:, -1] == 'M').astype(int).values

# Train/val/test split (60/20/20)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
for name, X_s, y_s in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
    print(f"{name} Accuracy: {accuracy_score(y_s, model.predict(X_s)):.4f}")

print("\nTest Classification Report:")
print(classification_report(y_test, model.predict(X_test), target_names=["Rock", "Mine"]))

