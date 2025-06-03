import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import to_device


# Evaluate model on test set and show metrics + confusion matrix
def evaluate(model, dataloader, model_type="Multimodal", show_matrix=True):
    model.eval()
    device = next(model.parameters()).device
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            if model_type == "Text":
                texts, _, labels = batch
                outputs = model(texts)
            else:
                texts, images, labels = batch
                outputs = model(texts, images)

            labels = to_device(labels, device)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute evaluation metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    # Print metrics
    print(f"[{model_type}] Accuracy:  {acc:.2f}")
    print(f"[{model_type}] Precision: {prec:.2f}")
    print(f"[{model_type}] Recall:    {rec:.2f}")
    print(f"[{model_type}] F1 Score:  {f1:.2f}")

    # Plot confusion matrix
    if show_matrix:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
        cmap = {"Multimodal": "Blues", "Text": "Oranges", "LateFusion": "Purples"}.get(model_type, "Greys")
        disp.plot(cmap=cmap)
        plt.title(f"{model_type} Model Confusion Matrix")
        plt.show()

    return acc, prec, rec, f1, cm
