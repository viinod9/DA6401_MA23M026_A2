import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet50_Weights, Inception_V3_Weights
import wandb
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(name, freeze_ratio, last_only=False):
    if name == 'resnet':
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)
    elif name == 'inception':
        weights = Inception_V3_Weights.DEFAULT
        model = models.inception_v3(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)

    if last_only:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        total = sum(1 for _ in model.children())
        freeze_layers = int(freeze_ratio * total)
        for idx, child in enumerate(model.children()):
            for param in child.parameters():
                param.requires_grad = idx >= freeze_layers

    return model

def prepare_data(data_path, augment=False):
    if augment:
        tfms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    else:
        tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    full_data = ImageFolder(data_path, transform=tfms)
    train_idx, val_idx = train_test_split(range(len(full_data)), test_size=0.2, random_state=42)
    train_loader = DataLoader(full_data, batch_size=32, sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(full_data, batch_size=32, sampler=SubsetRandomSampler(val_idx))
    return train_loader, val_loader

def run_training(model, train_dl, val_dl, num_epochs, cfg):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_acc_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_dl)
        train_acc = 100 * correct / total
        train_acc_list.append(train_acc)

        wandb.log({"Train Loss": avg_loss, "Train Accuracy": train_acc})

        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        val_acc = 100 * correct_val / total_val
        val_acc_list.append(val_acc)

        wandb.log({"Validation Accuracy": val_acc, "Epoch": epoch + 1})

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    print("Training finished.")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_acc_list, label="Train Accuracy", marker='o')
    plt.plot(range(1, num_epochs + 1), val_acc_list, label="Validation Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy Curve\narch={cfg.architecture}, freeze={cfg.freeze_ratio}, epochs={cfg.epochs}")
    plt.legend()
    plt.grid(True)

    filename = f"arch-{cfg.architecture}_freeze-{cfg.freeze_ratio}_ep-{cfg.epochs}.png"
    plt.savefig(filename)
    print(f"Saved accuracy plot as {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='resnet', choices=['resnet', 'inception'])
    parser.add_argument('--freeze_ratio', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--wandb_key', type=str, required=True)

    cfg = parser.parse_args()

    wandb.login(key=cfg.wandb_key)
    wandb.init(project="A2_PartB_finetune_args", config=vars(cfg))
    model = get_model(cfg.architecture, cfg.freeze_ratio)
    train_dl, val_dl = prepare_data(cfg.data_path, augment=cfg.augment)
    run_training(model, train_dl, val_dl, cfg.epochs, cfg)
    wandb.finish()

if __name__ == "__main__":
    main()
