from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
model.fit(X, y)

tree = model.tree_

features = tree.feature
thresholds = tree.threshold
impurity = tree.impurity
samples = tree.n_node_samples

feature_names = iris.feature_names

for i in range(tree.node_count):
    left = tree.children_left[i]
    right = tree.children_right[i]

    if left != -1:  # not a leaf
        parent_entropy = impurity[i]
        left_entropy = impurity[left]
        right_entropy = impurity[right]

        w_left = samples[left] / samples[i]
        w_right = samples[right] / samples[i]

        info_gain = parent_entropy - (w_left * left_entropy + w_right * right_entropy)

        print(f"\nNode {i}")
        print(f"Feature: {feature_names[features[i]]}")
        print(f"Threshold: {thresholds[i]:.2f}")
        print(f"Information Gain: {info_gain:.4f}")
        for name, importance in zip(feature_names, model.feature_importances_):
            print(f"{name}: {importance:.4f}")