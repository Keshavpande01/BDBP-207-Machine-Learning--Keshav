import pandas as pd
import numpy as np

# load dataset
df = pd.read_csv("sonar.csv", header=None)

# shape
print("Shape:", df.shape)

# first rows
print(df.head())

# column info
print(df.info())


#Separate Features and Target
X = df.iloc[:, :-1]   # first 60 columns
y = df.iloc[:, -1]    # last column

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Check missing values
print(df.isnull().sum())

#Statistical Summary
print(df.describe())

# check class Distrubution
print(y.value_counts())


# Feature relationship
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,10))
sns.heatmap(X.corr())
plt.title("Feature Correlation")
plt.show()

# Convert target to Numeric
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

print(y[:10])

#Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.3,
    random_state=42
)

# Train a Simple Model (Logistic Regression )
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))


# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)