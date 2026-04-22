### Try classifying classes 1 and 2 from the iris dataset with SVMs, with the 2 first features. Leave out 10% of each class and test prediction performance on these observations.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from RBF_Kernel import plot_decision_boundary

iris = load_iris()
X = iris.data
y = iris.target

mk = y > 0
X = X[mk][:, :2]  # X[mk]: filters only rows of class 1 and 2 , [:, :2]: selects only the 1st 2 features (sepal length & sepal width - both in cm).
y = y[mk]-1   # relabel class 1 to 0 and class 2 to 1 for binary classification

X_train, X_test, y_train, y_test = [], [], [], []   # to collect train/test sets for each class separately
for cls in [0, 1]:
    X_cls = X[y == cls]   # X_cls = X[y == cls]: filters data where the label is cls (0 or 1).
    y_cls = y[y == cls]
    X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_cls, test_size=0.1, random_state=42)   # 90% train set, 10% test set for each class
    X_train.extend(X_tr)   # .extend(): adds the new data to the combined train/test lists
    X_test.extend(X_te)
    y_train.extend(y_tr)
    y_test.extend(y_te)
X_train = np.array(X_train)   # convert lists to numpy arrays for model training and evaluation
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

svm_clf = SVC(kernel='linear', C=1)  # Train the SVM model
svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100} %")
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names[1:3]))    # target_names=...: gives readable class names ("versicolor", "virginica)

plot_decision_boundary(svm_clf, np.vstack((X_train, X_test)), np.hstack((y_train, y_test)), "SVM on Iris Dataset (Classes 1 & 2)")    # Combines training and testing data (for full view), and plots the decision boundary.
print(svm_clf.support_vectors_)

# Interpretation:
# We will get same classification output in all the three kernels - linear, RBF, polynomial, bec. we have small data size, since we are considering only 2 features here.
# SVM with a linear kernel does a decent job already. But the sepal-based features don't give enough discriminatory power for classes 1 vs 2.
# Hence the support vectors allow margin violations (especially with C=1).