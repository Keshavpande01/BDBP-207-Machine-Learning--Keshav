import numpy as np
import pandas as pd
from ISLP import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ------------------ LOAD DATA ------------------
df = load_data("OJ")

# Convert target variable
df['Purchase'] = df['Purchase'].map({'CH': 0, 'MM': 1})

# Convert categorical variable
df['Store7'] = df['Store7'].map({'No': 0, 'Yes': 1})

# One-hot encode STORE
df = pd.get_dummies(df, columns=['STORE'], drop_first=True)

# ------------------ FEATURES & TARGET ------------------
X = df.drop(columns=['Purchase'])
y = df['Purchase']

# ------------------ TRAIN TEST SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=800, random_state=42
)

# ------------------ SCALING ------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# (b) SVM (Linear Kernel → Soft Margin Classifier)
# =====================================================
svm_linear = SVC(kernel='linear', C=0.01)
svm_linear.fit(X_train, y_train)

train_acc_linear = accuracy_score(y_train, svm_linear.predict(X_train))
test_acc_linear = accuracy_score(y_test, svm_linear.predict(X_test))

print("Linear SVM:")
print("Train Accuracy:", train_acc_linear)
print("Test Accuracy :", test_acc_linear)

# =====================================================
# (c) SVM (RBF Kernel)
# =====================================================
svm_rbf = SVC(kernel='rbf')  # default gamma
svm_rbf.fit(X_train, y_train)

train_acc_rbf = accuracy_score(y_train, svm_rbf.predict(X_train))
test_acc_rbf = accuracy_score(y_test, svm_rbf.predict(X_test))

print("\nRBF Kernel SVM:")
print("Train Accuracy:", train_acc_rbf)
print("Test Accuracy :", test_acc_rbf)