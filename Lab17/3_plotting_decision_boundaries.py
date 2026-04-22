### Try this tutorial for plotting decision boundaries - https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm    # Provides support vector machine (SVM) classifiers
from sklearn.inspection import DecisionBoundaryDisplay    # A tool that helps visualize the decision boundary of classifiers

# Create a 2-D classification dataset with 16 samples and 2 classes -
X = np.array([
    [0.4, -0.7],
    [-1.5, -1.0],
    [-1.4, -0.9],
    [-1.3, -1.2],
    [-1.1, -0.2],
    [-1.2, -0.4],
    [-0.5, 1.2],
    [-1.5, 2.1],
    [1.0, 1.0],
    [1.3, 0.8],
    [1.2, 0.5],
    [0.2, -2.0],
    [0.5, -2.4],
    [0.2, -2.3],
    [0.0, -2.7],
    [1.3, 2.1],
])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

# Plotting settings -
fig, ax = plt.subplots(figsize=(4,3))     # plt.subplots() - creates a figure & a set of axes where data will be displayed.
x_min, x_max, y_min, y_max = -3, 3, -3, 3    # This sets the boundaries of the plot.
ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))    # This sets the x & y limits of the plot to ensure that all the plotted points & decision boundaries are within the range of -3 to 3.

# Plot the samples by color & add legend -
# s=150 - sets the size of each point to 150px, c=y - color the points based on their class labels, label=y - assigns labels to the points (used for legends), edgecolors="k" - draws black edges around each point.
#  X[:, 0] - extracts the 1st column of X (x-coordinate), X[:, 1] - extracts the 2nd column of X (y-coordinate)
scatter = ax.scatter(X[:, 0], X[:, 1], s=150, c=y, label=y, edgecolors="k")
ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")   # scatter.legend_elements() - generates the legend elements (one for each class in y
ax.set_title("Samples in 2-D feature space")
_ = plt.show()    # The samples are not clearly separable by a straight line.

# Training SVC model and plotting decision boundaries -
def plot_training_data_with_decision_boundary(kernel, ax=None, long_title=True, support_vectors=True):      # Inputs: kernel (type of kernel function), ax (optional; to specify axes for plotting)
    # Train the SVC -
    clf = svm.SVC(kernel=kernel, gamma=2).fit(X, y)    # gamma=2 - controls the influence of individual data points (for non-linear kernels).
    # Settings for plotting -
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))    # If 'ax' is not provided, then create a new figure & axes.
    x_min, x_max, y_min, y_max = -3, 3, -3, 3
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))   # otherwise, ensure that all the decision boundaries are plotted within this range.

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X, "ax": ax}
    # response_method="predict" - uses the classifier's predictions to color regions, plot_method="pcolormesh" - fills the decision boundary with colors, alpha=0.3 - sets transparency.
    DecisionBoundaryDisplay.from_estimator(**common_params, response_method="predict", plot_method="pcolormesh", alpha=0.3)
    # levels=[-1, 0, 1] - draws 3 contours: 0 (decision boundary), -1 and 1 (margins / support vectors), colors=["k","k","k"] - uses black lines, linestyles=["--","-","--"] - dashed margins & a solid boundary.
    DecisionBoundaryDisplay.from_estimator(**common_params, response_method="decision_function", plot_method="contour", levels=[-1, 0, 1], colors=["k", "k", "k"], linestyles=["--", "-", "--"])

    if support_vectors:
        # Plot bigger circles around samples that serve as support vectors
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=150, facecolors="none", edgecolors="k")    # returns support vectors (most imp. points for classification)

    # Plot samples by color and add legend
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    if long_title:
        ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")
    else:
        ax.set_title(kernel)
    if ax is None:
        plt.show()

def main():
    plot_training_data_with_decision_boundary("linear")
    plot_training_data_with_decision_boundary("poly")
    plot_training_data_with_decision_boundary("rbf")
    plot_training_data_with_decision_boundary("sigmoid")
    plt.show()

if __name__ == "__main__":
    main()