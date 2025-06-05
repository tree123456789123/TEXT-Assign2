import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Move tensor to specified device
def to_device(tensor, device):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    return tensor.to(device)


# Compute evaluation metrics
def compute_metrics(preds, labels):
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "confusion_matrix": confusion_matrix(labels, preds)
    }
