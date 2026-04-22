### Implement Adaboost Classifier from scratch using the iris dataset.

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data():
    data = pd.read_csv("Iris.csv")
    X = data.drop(columns=["Species", "Id"])   # Drop non-feature columns
    y = data['Species']   # Target variable (label)
    return X, y

def data_processing(X_train, X_test, y_train, y_test):
    encoder = LabelEncoder()
    y_encoded_train = encoder.fit_transform(y_train)
    y_encoded_test = encoder.transform(y_test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_encoded_train, y_encoded_test, encoder

class adaboost_classifier:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators   # No. of boosting rounds
        self.alphas = []   # List to store alpha values for each weak learner.
        self.models = []   # List to store weak learners,
        self.classes = None  # List of unique classes

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)  # Get unique class labels
        K = len(self.classes)  # Total no. of classes (there are 3 in iris dataset)
        w = np.ones(n_samples) / n_samples   # Initialize uniform sample weights

        for t in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)  # Weak learner (decision stump)
            model.fit(X, y, sample_weight=w)   # Train with current weights
            y_pred = model.predict(X)   # Predict on training data

            incorrect = (y_pred != y).astype(int)   # True if wrong - converts to 1 (misclassified), False if correct - converts to 0 (correctly classified).
            err_t = np.dot(w, incorrect) / np.sum(w)  # Computes the weighted error of the current weak learner.
            err_t = max(err_t, 1e-10)   # To avoid division by 0 or taking log of 0 in the next step, we cap the min. error to a small positive value (1e-10).
            if err_t >= 1 - 1e-10:   # If error is too high, stop boosting.
                break
            # Compute alpha (weight of weak learner)
            alpha_t = np.log((1 - err_t) / err_t) + np.log(K - 1)   # err_t: How bad the learner is , (1 - err_t) / err_t: Higher if the learner is good.

            # Update weights for next iteration -
            for i in range(len(w)):
                if incorrect[i] == 1:
                    w[i] *= np.exp(alpha_t)  # If a sample was misclassified, its weight increases — so it's more likely to be chosen next round.
                else:
                    w[i] *= np.exp(-alpha_t)  # If a sample was correctly classified, its weight decreases.
            # Normalize weights to maintain a probability distribution
            w /= np.sum(w)
            self.models.append(model)   # Store the trained weak learner (stump)
            self.alphas.append(alpha_t)  # Store alpha_t (its importance for final prediction)

    def predict(self, X):
        pred = np.zeros((X.shape[0], len(self.classes)))   # Creates a 2D array to accumulate scores for each class. Each cell will store the sum of alpha values for that class.
        for alpha, model in zip(self.alphas, self.models):  # Iterates over each trained model and its alpha.
            y_pred = model.predict(X)   # Predicts the class for all the samples.
            pred[np.arange(X.shape[0]), y_pred] += alpha  # Adds that model’s weight (alpha) to the predicted class score for each sample.
        return self.classes[np.argmax(pred, axis=1)]   # gets the index (class) with the highest vote score ; self.classes[] maps that index back to the original class label

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=4343)
    X_train_scaled, X_test_scaled, y_train, y_test, y_encoder = data_processing(X_train, X_test, y_train, y_test)

    print("\nScikit-learn AdaBoost Classifier:")
    weak_learner = DecisionTreeClassifier(max_depth=1)
    adaboost = AdaBoostClassifier(estimator=weak_learner, n_estimators=50, random_state=42)
    adaboost.fit(X_train_scaled, y_train)
    y_pred_sklearn = adaboost.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_sklearn, target_names=y_encoder.classes_))

    print("\nCustom AdaBoost Classifier:")
    custom_adaboost = adaboost_classifier(n_estimators=50)
    custom_adaboost.fit(X_train_scaled, y_train)
    y_pred_custom = custom_adaboost.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_custom, target_names=y_encoder.classes_))

if __name__ == "__main__":
    main()