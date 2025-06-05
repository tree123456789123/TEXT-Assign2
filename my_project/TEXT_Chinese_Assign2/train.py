import torch
from tqdm import tqdm
from utils import to_device


# General training loop for all models
def train_general(model, train_loader, epochs, lr, save_prefix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"[{save_prefix}] Epoch {epoch + 1}")
        for texts, images, labels in loop:
            labels = to_device(labels, device)

            # Forward pass based on model type
            if save_prefix == "Text":
                outputs = model(texts)
            else:
                outputs = model(texts, images)

            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / (loop.n or 1))

        avg_loss = running_loss / len(train_loader)
        print(f"[{save_prefix}] Epoch {epoch + 1} Loss: {avg_loss:.4f}")

        # Save model after each epoch
        torch.save(model, f"{save_prefix.lower()}_model_epoch{epoch + 1}.pth")
