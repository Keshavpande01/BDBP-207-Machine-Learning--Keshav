from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target.reshape(-1, 1)


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Model
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Feature importance
print("\nFeature Importance:")
for i, v in enumerate(model.feature_importances_):
    print(f"Feature {i}: {v:.3f}")

# Tree visualization
plt.figure(figsize=(15,10))
plot_tree(model, filled=True)
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

params = {
    "max_depth": [2, 3, 4, 5,6, None],
    "min_samples_split": [2,3, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(
    DecisionTreeClassifier(criterion='entropy', random_state=42),
    params,
    cv=5
)
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("CV Score:", grid.best_score_)

