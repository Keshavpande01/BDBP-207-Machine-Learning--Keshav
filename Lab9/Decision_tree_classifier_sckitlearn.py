import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load dataset
iris = load_iris()

# Take 2 features → Petal Length & Petal Width
X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model on training data
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))

# Create mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

# Predict for each grid point
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))

# Background regions
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

# Scatter points
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.RdYlBu)

# Labels
# Train points → blue circles
colors = ['red', 'green', 'blue']

for i, color in enumerate(colors):
    plt.scatter(X_train[y_train == i, 0],
                X_train[y_train == i, 1],
                color=color, marker='o', label=f"Train Class {i}")

    plt.scatter(X_test[y_test == i, 0],
                X_test[y_test == i, 1],
                color=color, marker='x', s=100, label=f"Test Class {i}")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Decision Tree Decision Boundary (Iris)")
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=iris.target_names))
# Legend
plt.legend()

plt.show()