import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data loading and cleaning
df = pd.read_csv("data.csv")

# Drop ID column
df = df.drop("id", axis=1)

# Convert diagnosis to numeric
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Feature and target split
X =df.drop("diagnosis", axis=1)
y = df["diagnosis"]


# Train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=4, random_state=42)


# Model training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

#Feature importance
importance = model.feature_importances_

for i, v in enumerate(importance):
    print(f"Feature {i}: {v}")

# Tree visualization
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))
plot_tree(model, filled=True)
plt.show()