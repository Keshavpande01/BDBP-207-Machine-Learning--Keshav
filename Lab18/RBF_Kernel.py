### Implement the RBF kernel. Check if RBF kernel separates the data well and compare it with the Polynomial Kernel.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Define the dataset -
X = np.array([[6, 5], [6, 9], [8, 6], [8, 8], [8, 10], [9, 2], [9, 5], [10, 10], [10, 13], [11, 5], [11, 8], [12, 6], [12, 11], [13, 4], [14, 8]])
# Define the labels -
labels = ["Blue", "Blue", "Red", "Red", "Red", "Blue", "Red", "Red", "Blue", "Red", "Red", "Red", "Blue", "Blue", "Blue"]
# Encode labels
le = LabelEncoder()   # creates an encoder object
y = le.fit_transform(labels)   # Blue = 0, Red = 1

# Gaussian RBF kernel is generally applied when training set is not too large.
# SVC(): creates SVM classifier models, rbf: radial basis function (nonlinear), gamma='scale': automatically sets hyperparameter 'gamma' based on data variance
# C=1: regularization parameter (controls margin vs error trade-off).
rbf_svm = SVC(kernel="rbf", gamma='scale', C=1)
# Increasing 'gamma' makes the bell-shape curve narrower & as a result each instance’s range of influence is smaller: the decision boundary ends up being more irregular, wiggling around individual instances.
# So 'gamma' acts like a regularization hyperparameter: if the model is overfitting, then we should reduce it, and if it is underfitting, we should increase it (similar to the C hyperparameter).

# The hyperparameter 'coef0' controls how much the model is influenced by high degree polynomials versus low-degree polynomials.
poly_svm = SVC(kernel="poly", degree=3, coef0=1, C=1)    # If the model is overfitting, then reduce the polynomial degree.
# Training each SVM model on given dataset -
rbf_svm.fit(X, y)
poly_svm.fit(X, y)

def plot_decision_boundary(clf, X, y, title):
    # Input: clf - trained model, X - 2D points (x1, x2), y - class labels (0 or 1), title - title for the plot.
    h = 0.2   # Sets the step size for the grid (higher - faster but coarse, lower - slow but detailed); controls how fine the decision boundary looks.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1   # X[:, 0] - all x1 values, -1 & +1 - gives a little padding around plot edges so points don't touch the border.
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1   # for x2 values
    # np.arange(...) creates a range of values from min. to max. with step h , np.meshgrid() turns those ranges into two 2D grids: one for x and one for y.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))   # So, we're simulating a grid of points over the entire 2D space.
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])    # xx.ravel() & yy.ravel() flatten the 2D grid into 1D arrays , clf.predict() predicts the class label for each point in the grid.
    Z = Z.reshape(xx.shape)   # reshapes the predictions to match the shape of the meshgrid, so that we can plot it as a 2D surface.
    plt.figure(figsize=(6, 5))   # 6 inches wide, 5 inches tall
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)   # cmap=plt.cm.coolwarm: uses red-blue colour map , alpha=0.3: makes it partially transparent so that we can see the points underneath.
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=60)   # c=y: color the x1 & x2 values based on their true class label, s=60: sets the size of the points.
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

def main():
    plot_decision_boundary(rbf_svm, X, y, "SVM with RBF Kernel")
    plot_decision_boundary(poly_svm, X, y, "SVM with Polynomial Kernel (degree=3)")

if __name__ == "__main__":
    main()

# Interpretation:
# RBF Kernel: Smooth & circular boundary, very high flexibility, handles non-linear data very well, best fit for this dataset, overfitting risk is low (with proper gamma).
# Polynomial Kernel: Angular & less smooth boundary, moderate flexibility, handles non-linear data but depends on degree, slightly less precise for this dataset, slightly high risk of overfitting (depends on degree & C).