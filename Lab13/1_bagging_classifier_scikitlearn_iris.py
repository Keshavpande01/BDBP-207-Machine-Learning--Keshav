### Implement bagging classifier using scikit-learn. Use iris dataset from scikit-learn.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()    # loading the iris dataset
X, y = iris.data, iris.target    # X: features, y: target
feature_names = iris.feature_names    # column names
df = pd.DataFrame(X, columns=feature_names)    # converting into dataframe

# EDA -
print("Dataset Overview: ")
print(df.head())
print(df.info())
print("Summary Statistics: ")
print(df.describe())
print("Unique values: ")
print(df.nunique())
print("Column names: ")
print(df.columns)
print("Is there any missing (null) values present? ")
print(df.isnull().sum())
print("Is there any duplicate values present? ")
print(df.duplicated().sum())

# Split the dataset into 70% train set & 30% test set -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling -
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the model -
bag_clf = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5), n_estimators=100, max_features=1.0, max_samples=1.0, bootstrap_features=False, bootstrap=True, random_state=42, n_jobs=-1)
bag_clf.fit(X_train_scaled, y_train)
y_pred = bag_clf.predict(X_test_scaled)
print("Predicted y-values: ")
print(y_pred)

# Evaluate the model -
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy*100,"%")
class_report = classification_report(y_test, y_pred)
print("Classification Report: ")
print(class_report)