# =========================================================
# ViT-B/16 Training for Non-invasive Anemia Detection
# Structured & Paper-aligned Implementation
# =========================================================

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
class Config:
    TRAIN_DIR = "/home/khushi/Pixonate/Vit_conjuctiva/data_splitted/train"
    TEST_DIR = "/home/khushi/Pixonate/Vit_conjuctiva/data_splitted/test"
    SAVE_DIR = "checkpoints"
    BATCH_SIZE = 8
    EPOCHS = 30
    LR = 1e-5
    NUM_CLASSES = 2
    PATIENCE = 6
    CHECKPOINT_EVERY = 5
    IMG_SIZE = 224

# -------------------------
# DEVICE
# -------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# TRANSFORMS
# -------------------------
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    return train_tf, test_tf

# -------------------------
# DATA LOADERS
# -------------------------
def get_dataloaders():
    train_tf, test_tf = get_transforms()
    train_ds = datasets.ImageFolder(Config.TRAIN_DIR, transform=train_tf)
    test_ds = datasets.ImageFolder(Config.TEST_DIR, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    print("Class mapping:", train_ds.class_to_idx)
    return train_loader, test_loader

# -------------------------
# MODEL
# -------------------------
def build_model(device):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=Config.NUM_CLASSES
    )
    return model.to(device)

# -------------------------
# TRAIN ONE EPOCH
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

# -------------------------
# VALIDATE
# -------------------------
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds, gts = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs.logits, labels)
            running_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=1)
            preds.extend(predictions.cpu().numpy())
            gts.extend(labels.cpu().numpy())

    acc = accuracy_score(gts, preds)
    return running_loss / len(loader), acc

# -------------------------
# TRAINING PIPELINE
# -------------------------
def train():
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    device = get_device()
    print("Using device:", device)

    train_loader, test_loader = get_dataloaders()
    model = build_model(device)

    optimizer = torch.optim.NAdam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch [{epoch+1}/{Config.EPOCHS}]")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(
            model, test_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # ---- Best model ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{Config.SAVE_DIR}/best_model.pth")
            print("✅ Best model saved")
        else:
            patience_counter += 1
            print(f"Early stopping: {patience_counter}/{Config.PATIENCE}")

        # ---- Checkpoint ----
        if (epoch + 1) % Config.CHECKPOINT_EVERY == 0:
            torch.save(
                model.state_dict(),
                f"{Config.SAVE_DIR}/checkpoint_epoch_{epoch+1}.pth"
            )

        # ---- Early stopping ----
        if patience_counter >= Config.PATIENCE:
            print("🛑 Early stopping triggered")
            break

    plot_losses(train_losses, val_losses)

# -------------------------
# PLOT LOSSES
# -------------------------
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    train()