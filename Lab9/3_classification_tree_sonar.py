
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# Data loading and cleaning
data=pd.read_csv("sonar.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Convert Rock/Metal to 0/1
y = np.where(y == 'M', 1, 0)
print(data.shape)

# Train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)


# Model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(class_weight='balanced', random_state=42)


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

print("\nClassification Report:\n")
print(classification_report(y_test,y_pred, target_names=["Rock","Mine"]))
