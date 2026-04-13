### Implement Random Forest algorithm for classification using scikit-learn. Use iris datasets from scikit-learn.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()    # loading the iris dataset
X, y = iris.data, iris.target    # X: features, y: target
feature_names = iris.feature_names    # column names
df = pd.DataFrame(X, columns=feature_names)    # converting into dataframe

# EDA already done in excerise-1.1

# Split the dataset into 70% train set & 30% test set -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling -
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the model -
random_forest_clf = RandomForestClassifier(criterion='entropy', max_depth=3, n_estimators=100, max_features='sqrt', bootstrap=True, random_state=42, n_jobs=-1)
random_forest_clf.fit(X_train_scaled, y_train)
y_pred = random_forest_clf.predict(X_test_scaled)
print("Predicted y-values: ")
print(y_pred)

# Evaluate the model -
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy*100,"%")
class_report = classification_report(y_test, y_pred)
print("Classification Report: ")
print(class_report)


