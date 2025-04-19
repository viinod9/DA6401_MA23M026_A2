
import kagglehub
viinod9_inaturalist_dataset_path = kagglehub.dataset_download('viinod9/inaturalist-dataset')

print('Data source import complete.')


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet50_Weights, Inception_V3_Weights
import wandb  # for logging and visualization

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to get the pretrained model with some layers frozen
def get_model(name, freeze_ratio, last_only=False):
    # If architecture is ResNet
    if name == 'resnet':
        weights = ResNet50_Weights.DEFAULT  # Load default pretrained weights
        model = models.resnet50(weights=weights)  # Load pretrained ResNet50 model
        num_features = model.fc.in_features  # Get number of input features to final FC layer
        model.fc = nn.Linear(num_features, 10)  # Replace final FC layer for 10 classes

    # If architecture is Inception
    elif name == 'inception':
        weights = Inception_V3_Weights.DEFAULT
        model = models.inception_v3(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)  # Replace final FC layer for 10 classes

    # Option to freeze all layers except the final one
    if last_only:
        for param in model.parameters():
            param.requires_grad = False  # Freeze all layers
        for param in model.fc.parameters():
            param.requires_grad = True  # Unfreeze final FC layer only
    else:
        # Freeze only a certain ratio of layers from the beginning
        total = sum(1 for _ in model.children())  # Count total layers
        freeze_layers = int(freeze_ratio * total)  # Number of layers to freeze
        for idx, child in enumerate(model.children()):
            for param in child.parameters():
                param.requires_grad = idx >= freeze_layers  # Freeze first part, unfreeze the rest

    return model  # Return customized model


# Function to load and preprocess the dataset
def prepare_data(data_path, augment=False):
    # Apply data augmentation if specified
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
        # Simple resizing and normalization for validation
        tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    # Load dataset from folders
    full_data = ImageFolder(data_path, transform=tfms)

    # Split indices for training and validation (80-20 split)
    train_idx, val_idx = train_test_split(range(len(full_data)), test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = DataLoader(full_data, batch_size=32, sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(full_data, batch_size=32, sampler=SubsetRandomSampler(val_idx))
    return train_loader, val_loader

import matplotlib.pyplot as plt

# Function to train the model and validate it after each epoch
def run_training(model, train_dl, val_dl, num_epochs, cfg):
    model.to(device)  # Move model to GPU if available
    loss_fn = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    #optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_acc_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        correct = 0
        total = 0

        # Iterate over training batches
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients
            outputs = model(imgs)  # Forward pass
            loss = loss_fn(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)  # Get class predictions
            correct += (preds == labels).sum().item()  # Count correct predictions
            total += labels.size(0)

        avg_loss = total_loss / len(train_dl)
        train_acc = 100 * correct / total  # Training accuracy
        train_acc_list.append(train_acc)

        # Log training metrics to WandB
        wandb.log({"Train Loss": avg_loss, "Train Accuracy": train_acc})

        # Validation loop (no gradient updates)
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
        val_acc = 100 * correct_val / total_val  # Validation accuracy
        val_acc_list.append(val_acc)

        # Log validation metrics to WandB
        wandb.log({"Validation Accuracy": val_acc, "Epoch": epoch + 1})

        # Print training progress
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    print("Training finished.")

    # Plot and save the accuracy graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_acc_list, label="Train Accuracy", marker='o')
    plt.plot(range(1, num_epochs + 1), val_acc_list, label="Validation Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy Curve\narch={cfg.architecture}, freeze={cfg.freeze_ratio}, epochs={cfg.epochs}")
    plt.legend()
    plt.grid(True)

    # Save the accuracy plot as an image
    filename = f"arch-{cfg.architecture}_freeze-{cfg.freeze_ratio}_ep-{cfg.epochs}.png"
    plt.savefig(filename)
    print(f"Saved accuracy plot as {filename}")


# WandB sweep configuration for trying different hyperparameters automatically
sweep_configuration = {
    'method': 'bayes',
    'metric': {'name': 'Validation Accuracy', 'goal': 'maximize'},
    'parameters': {
        'epochs': {'values': [10, 15, 20]},
        'freeze_ratio': {'values': [0.8]},
        'architecture': {'values': ['resnet']},
        'learning_rate': {'values': [0.001, 0.0001, 0.01]},
        'augment': {'values': [True, False]}
    }
}


# Login to Weights & Biases using your API key
wandb.login(key="acdc26d2fc17a56e83ea3ae6c10e496128dee648")
# Initialize sweep in WandB with given configuration
sweep_id = wandb.sweep(sweep=sweep_configuration, project="A2_PartB_finetune_bayes")




def load_test_data(val_dir, batch_size=32):
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    test_data = ImageFolder(val_dir, transform=tfms)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    class_names = test_data.classes
    return test_loader, class_names


def predict_and_log_images(model, test_loader, class_names):
    model.eval()
    all_images = []
    count_per_class = {cls: 0 for cls in class_names}
    max_per_class = 3  # Only 3 per class
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            for img, pred, actual in zip(imgs, preds, labels):
                pred_class = class_names[pred.item()]
                actual_class = class_names[actual.item()]
                if count_per_class[actual_class] < max_per_class:
                    all_images.append(wandb.Image(img.cpu(), caption=f"Pred: {pred_class} | Actual: {actual_class}"))
                    count_per_class[actual_class] += 1
            if all(count >= max_per_class for count in count_per_class.values()):
                break
    wandb.log({"Predicted vs Actual (10x3)": all_images})


def sweep_runner():
    with wandb.init() as run:
        cfg = wandb.config
        wandb.run.name = f"arch-{cfg.architecture}_freeze-{cfg.freeze_ratio}_ep-{cfg.epochs}_lr-{cfg.learning_rate}_aug-{cfg.augment}"

        model = get_model(cfg.architecture, cfg.freeze_ratio, last_only=False)
        data_dir = "/kaggle/input/inaturalist-dataset/inaturalist_12K/train"
        val_dir = "/kaggle/input/inaturalist-dataset/inaturalist_12K/val"

        train_dl, val_dl = prepare_data(data_dir, augment=cfg.augment)
        run_training(model, train_dl, val_dl, num_epochs=cfg.epochs, cfg=cfg)

        # 🔽 Evaluate on test set
        test_loader, class_names = load_test_data(val_dir)
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        test_acc = 100 * correct / total
        print(f"Test Accuracy: {test_acc:.2f}%")
        wandb.log({"Test Accuracy": test_acc})

        # Log image grid
        predict_and_log_images(model, test_loader, class_names)



# sweep_config = {
#     'method': 'random',  # or 'grid', 'bayes'
#     'metric': {'name': 'Validation Accuracy', 'goal': 'maximize'},
#     'parameters': {
#         'epochs': {'values': [1]},
#         'freeze_ratio': {'values': [0.8]},
#         'architecture': {'values': ['resnet']},
#         'learning_rate': {'values': [0.001]},
#         'augment': {'values': [True]}
#     }
# }




# sweep_id = wandb.sweep(sweep_config, project="transfer_learning_sweep")
# wandb.agent(sweep_id, function=sweep_runner, count=1)  # or count=None for indefinite