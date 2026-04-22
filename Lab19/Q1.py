### Implement a python code to calculate the following metrics: - accuracy, precision, sensitivity, specificity, F1-score, plot the ROC curve, AUC.

# This code is for testing the binary-classification models only.

import matplotlib.pyplot as plt

def compute_confusion_matrix(actual_labels, predicted_probs, thresholds):
    results = {}
    for threshold in thresholds:   # For each threshold
        TP = FP = TN = FN = 0    # Initializing all these to zero
        for actual, pred_prob in zip(actual_labels, predicted_probs):   # zip() - picks each actual label with its corresponding predicted label
            pred_label = 1 if pred_prob >= threshold else 0    # if predicted score > threshold , then it will be labelled as 1 , otherwise 0
            if actual == 1 and pred_label == 1:   # if actual label is positive and predicted label is positive, then it will be considered as true positive
                TP += 1
            elif actual == 0 and pred_label == 1:    # if actual label is negative and predicted label is positive, then it will be considered as false positive
                FP += 1
            elif actual == 0 and pred_label == 0:   # if actual label is negative and predicted label is negative, then it will be considered as true negative
                TN += 1
            elif actual == 1 and pred_label == 0:    # if actual label is positive and predicted label is negative, then it will be considered as false negative
                FN += 1
        results[threshold] = (TP, FP, TN, FN)
    return results

def calculate_accuracy(TP, TN, FP, FN):
    accuracy = (TP + TN)/ (TP+TN+FP+FN)
    return accuracy

def calculate_precision(TP, FP):
    return TP/(TP + FP) if (TP+FP) > 0 else 0    # bec. we don't want the denominator to be negative or zero

def calculate_sensitivity(TP, FN):
    return TP / (TP + FN) if (TP+FN) > 0 else 0

def calculate_specificity(TN, FP):
    return TN / (TN + FP) if (TN+FP) > 0 else 0

def calculate_F1_score(precision, TPR):
    return 2 * (precision * TPR) / (precision + TPR) if (precision + TPR) > 0 else 0

def roc_curve(actual_labels, predicted_probs, thresholds):
    fpr_val = []
    tpr_val = []
    for threshold in thresholds:
        TP, FP, TN, FN = compute_confusion_matrix(actual_labels, predicted_probs, [threshold])[threshold]
        TPR = calculate_sensitivity(TP, FN)   # True positive rate
        FPR = 1-(calculate_specificity(TN, FP))   # False Positive Rate
        tpr_val.append(TPR)
        fpr_val.append(FPR)
    return fpr_val, tpr_val

def compute_auc(fpr_val, tpr_val):
    auc = 0
    for i in range(1, len(fpr_val)):   # it will iterate over each threshold and will substitute the FPR value in the formula
        auc += ((fpr_val[i] - fpr_val[i - 1]) * (tpr_val[i] + tpr_val[i - 1])) / 2     # AUC Formula ; where, i = threshold
    return auc

def plot_roc_curve(fpr_val, tpr_val):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_val, tpr_val, linestyle='-', color='b', label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label="Reference Line")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()