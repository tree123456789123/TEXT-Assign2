import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def evaluate_model(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, images, labels in dataloader:
            labels = torch.tensor(labels).to(device)
            outputs = model(texts, images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy:  {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall:    {rec:.2f}")
    print(f"F1 Score:  {f1:.2f}")

    # Show confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
    disp.plot(cmap="Blues")
    plt.title("Multimodal Model Confusion Matrix")
    plt.show()

    return acc, prec, rec, f1, cm


def evaluate_text_model(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, _, labels in dataloader:
            labels = torch.tensor(labels).to(device)
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    acc_text = accuracy_score(all_labels, all_preds)
    prec_text = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"[Text] Accuracy:  {acc_text:.2f}")
    print(f"[Text] Precision: {prec_text:.2f}")
    print(f"[Text] Recall:    {rec:.2f}")
    print(f"[Text] F1 Score:  {f1:.2f}")

    # Show confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
    disp.plot(cmap="Oranges")
    plt.title("Text-Only Model Confusion Matrix")
    plt.show()

    return acc_text, prec_text, rec, f1, cm
