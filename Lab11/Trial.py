import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


### Implement decision tree classifier without using scikit-learn using the iris dataset. Fetch the iris dataset from scikit-learn library.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset -
iris = load_iris()
X = iris.data
y = iris.target

def count_classes(y):   # to count the no. of each label
    counts={}   # empty dictionary to store label counts
    for label in y:   # loops through each label in y
        if label in counts:  # if already in dict, increment count
            counts[label] += 1
        else:
            counts[label] = 1   # if not seen before, initialize count to 1
    return counts

def entropy(y):
    counts = count_classes(y)   # Get how many times each label occurs.
    n = sum(counts.values())   # Total no. of samples
    probabilities = [count/n for count in counts.values()]   # Compute each class's probability.
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p>0])   # Formula: H = -summation(p*log2(p))
    return entropy

def weighted_entropy(left_y,right_y):
    n_total = len(left_y) + len(right_y)   # Total no. of samples in left and right splits.
    left_total = len(left_y)
    right_total = len(right_y)
    H_left = entropy(left_y)   # Compute entropy of left split.
    H_right = entropy(right_y)   # Compute entropy of right split.
    return (H_left * (left_total)/n_total) + (H_right * (right_total)/n_total)   # Weighted average of both entropies.

def info_gain(parent_y,left_y,right_y):
    H_parent = entropy(parent_y)   # Entropy before split
    H_weighted = weighted_entropy(left_y,right_y)  # After split
    ig = H_parent - H_weighted  # Info. Gain: How much uncertainty was reduced.
    return ig

def best_split(X,y):   # best split decision based on max info. gain
    best_feature, best_threshold, best_ig = None,None,-1   # Initialization

    for feature_index in range(X.shape[1]):   # Loop through all the features (X.shape[1]: no. of columns)
        thresholds = np.unique(X[:,feature_index])

        # Get all unique values as potential split thresholds.

        for threshold in thresholds:   # all threshold points will be tested
            left_indices = X[:,feature_index] < threshold   # where values is less than threshold
            right_indices = ~left_indices   # =~ : logical NOT (those not in left)
            left_y,right_y = y[left_indices], y[right_indices]   # Extracts the corresponding labels (y) for the left and right splits.
            if len(left_y) > 0 and len(right_y) > 0:   # Ensures both splits have data; avoids empty splits.
                current_ig = info_gain(y, left_y, right_y)   # calculating info. gain for the current split
                if current_ig > best_ig:   # If the current IG is better than the best seen so far, update the best values.
                    best_ig = current_ig
                    best_feature, best_threshold = feature_index, threshold
    return best_feature, best_threshold   # Returns the best feature index and threshold to split the data.

def build_tree(X, y, max_depth=None, depth=0):   # max_depth: stopping condition, depth=0: current depth of the tree
    if max_depth is not None and depth == max_depth:    # If max_depth is reached, return the most common label as a leaf node.
        return Counter(y).most_common(1)[0][0]
    if len(set(y)) == 1:   # If all labels are the same, return the label.
        return y[0]
    feature_index, threshold = best_split(X, y)   # Finds the best feature and threshold to split the data.
    if feature_index is None:
       return Counter(y).most_common(1)[0][0]   # If no valid split found, return the majority class label.
    left_X,right_X = [], []   # Creates empty lists for left and right splits of both X and y.
    left_y,right_y = [], []
    for i in range(len(X)):   # Split the data based on the threshold.
        if X[i][feature_index] < threshold:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])
    left_X, right_X = np.array(left_X), np.array(right_X)   # Converts the split lists back into numpy arrays for further processing.
    left_y, right_y = np.array(left_y), np.array(right_y)
    # Recursively builds the left and right subtrees, increasing depth each time.
    left_subtree=build_tree(left_X,left_y,max_depth,depth+1)
    right_subtree=build_tree(right_X,right_y,max_depth,depth+1)
    return{
        'feature_index': feature_index,
        'threshold': threshold,
        'left': left_subtree,
        'right': right_subtree,
    }

def predict(tree,inputs):    # Predicts the class for a single sample.
    if not isinstance(tree, dict):   # If it's a leaf node (not a dict), return the class directly.
        return tree
    if inputs[tree['feature_index']] < tree['threshold']:   # Traverse left or right subtree depending on the threshold.
        return predict(tree['left'],inputs)
    else:
        return predict(tree['right'],inputs)

def predict_all(tree,X):   # Predicts labels for all samples in the test set.
    return np.array([predict(tree, inputs) for inputs in X])

# Split the dataset into 70% train set and 30% test set -
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.30,random_state=42)

tree = build_tree(X_train, y_train, max_depth=5)   # Builds the decision tree with max depth of 5.
y_pred = predict_all(tree, X_test)   # Predicts labels for the test data using the trained tree.
accuracy = np.sum(y_pred == y_test) / len(y_test)   # Compares predicted and actual labels, and counts correct predictions and divides by total samples.
print(f"Accuracy (modeled from scratch): {accuracy:.2f}")

# comparing it with Decision Tree Classifier in sklearn
count= {int(k):v for k, v in count_classes(y_pred).items()}
print(count)
clf_skl = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_skl.fit(X_train,y_train)
y_pred_skl = clf_skl.predict(X_test)
accuracy_skl =accuracy_score(y_test, y_pred_skl)
print(f"Accuracy (from sklearn): {accuracy:.2f}")