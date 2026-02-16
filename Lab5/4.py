### Implement logistic regression using scikit-learn for the breast cancer dataset - https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# Loading the dataset -
data = pd.read_csv("data.csv")

# EDA -
print("Overview: ")
print(data.info())
print(data.head())
print("Missing / null values: ")
print(data.isnull().sum())
print("Duplicate rows: ")
print(data.duplicated().sum())
print("Statistical summary: ")
print(data.describe())
print("Target variable distribution: ")    # Counting the no. of malignant & benign samples
print(data['diagnosis'].value_counts())
data = data.drop(['id', 'Unnamed: 32'], axis=1)    # Removing the columns that are not relevant for analysis

# One hot encoding -
encoder = OneHotEncoder(sparse_output=False, drop='first')    # Creating the OneHotEncoder instance
diagnosis_reshaped = data['diagnosis'].values.reshape(-1, 1)    # Reshape it to 2D
encoded = encoder.fit_transform(diagnosis_reshaped)
data['Malignant'] = encoded[:, 0]    # Adding the encoded column back to the dataset
data = data.drop('diagnosis', axis=1)    # Dropping the original column
print(data.head())

# Correlation analysis -
corr = data.corr().T
plt.figure(figsize=(35, 25))
sns.heatmap(corr,annot=True,cmap='mako_r')
plt.show()     # Almost all the features are highly correlated

# Outlier detection -
plt.figure(figsize=(40,10))
sns.boxplot(data=data)
plt.show()     # outliers - area_mean, area_worst

# Visualization of feature distribution -
histogram = data.hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.subplots_adjust(hspace=0.7, wspace=0.4)
plt.plot()
plt.show()

# Z-score normalization -
y = data['Malignant']
print(y)
data = data.drop('Malignant', axis=1)
X = (data - data.mean()) / data.std()
print(X.head())

# Splitting the data into 70% train set and 30% test set -
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=1000)      # max_iter to ensure convergence
log_reg.fit(X_train, y_train)    # Train the model
y_pred = log_reg.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score
accuracy = (accuracy_score(y_test, y_pred)) * 100
print("Accuracy: ", accuracy, "%")
errors = y_pred - y_test
std_deviation = np.std(errors)
print("Standard deviation: ", std_deviation)

# Visualization of the sigmoidal curve -
raw_predictions = log_reg.decision_function(X_test)     # z = X.theta + b (b=bias term); linear combination of the input features after being passed via the model
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
predicted_probabilities = sigmoid(raw_predictions)      # logistic function that maps the raw predictions to the probabilities, i.e., converts these b/w 0 & 1

sorted_indices = np.argsort(raw_predictions)      # sorts the indices of raw predictions in ascending order (for smooth curve)
sorted_probabilities = predicted_probabilities[sorted_indices]      # to reorder the predicted probabilities acc. to sorted order of raw ones
sorted_raw_predictions = raw_predictions[sorted_indices]

plt.figure(figsize=(8, 6))
plt.plot(sorted_raw_predictions, sorted_probabilities, color='blue')
plt.title('Logistic Regression Curve')
plt.xlabel('Raw predictions (z)')
plt.ylabel('Predicted probability (g(z)) - Malignant=1')
plt.show()