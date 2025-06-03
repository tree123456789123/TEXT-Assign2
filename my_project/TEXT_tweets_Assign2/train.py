import torch
from tqdm import tqdm


def train_model(model, train_loader, epochs=2, lr=1e-6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for texts, images, labels in loop:
            labels = torch.tensor(labels).to(device)

            # Forward pass
            outputs = model(texts, images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / (loop.n if loop.n else 1))

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")
        torch.save(model, f"multimodal_model_epoch{epoch + 1}.pth")


def train_textonly_model(model, train_loader, epochs=2, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"[Text] Epoch {epoch + 1}")
        for texts, _, labels in loop:
            labels = torch.tensor(labels).to(device)

            outputs = model(texts)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / (loop.n if loop.n else 1))

        print(f"[Text] Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")
        torch.save(model, f"textonly_model_epoch{epoch + 1}.pth")
