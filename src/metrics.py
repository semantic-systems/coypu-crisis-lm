import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def classifier_metrics(p):
    logits, labels = p
    preds = np.argmax(logits, axis=1)
    average = "macro"
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds, average=average)
    precision = precision_score(y_true=labels, y_pred=preds, average=average)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    f1_weighted = f1_score(y_true=labels, y_pred=preds, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "f1_weighted": f1_weighted}