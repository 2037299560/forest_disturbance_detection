from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, cohen_kappa_score
from tabulate import tabulate
import numpy as np

def calculate_model_accuracy(ground_truth, predictions):
    classes = np.unique(ground_truth)
    cm = confusion_matrix(ground_truth, predictions)
    acc = accuracy_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions, average=None)
    precision = precision_score(ground_truth, predictions, average=None)
    f1 = f1_score(ground_truth, predictions, average=None)
    kappa = cohen_kappa_score(ground_truth, predictions)

    table_data = []
    headers = ["Class", "Precision", "Recall", "F1 Score"]
    for i, class_label in enumerate(classes):
        table_data.append([int(class_label), precision[i], recall[i], f1[i]])
    table_str = tabulate(table_data, headers=headers, tablefmt="github", floatfmt=".4f")

    return {
        "confusion_matrix": cm,
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "kappa": kappa,
        "table_str": table_str
    }

