import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import wandb

# Custom CNN Definition
class Cutomized_CNN(nn.Module):
    def __init__(self, input_channels, conv_filters, kernel_sizes, activation_fn,
                 dense_neurons, dense_activation_fn, dropout, batch_norm, num_classes):
        super(Cutomized_CNN, self).__init__()
        self.activation_fn = activation_fn
        self.dense_activation_fn = dense_activation_fn
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        for out_channels, kernel_size in zip(conv_filters, kernel_sizes):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv_layers.append(nn.Sequential(*layers))
            in_channels = out_channels

        self._dummy_input = torch.zeros(1, input_channels, 224, 224)
        self.flattened_size = self._get_flattening_size()
        fc1_layers = [nn.Dropout(dropout)] if dropout > 0 else []
        fc1_layers.append(nn.Linear(self.flattened_size, dense_neurons))
        self.fc1 = nn.Sequential(*fc1_layers)
        self.fc2 = nn.Linear(dense_neurons, num_classes)

    def _get_flattening_size(self):
        x = self._dummy_input
        for block in self.conv_layers:
            for layer in block:
                x = layer(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        for block in self.conv_layers:
            for layer in block:
                x = layer(x) if not isinstance(layer, nn.Conv2d) else self.activation_fn(layer(x))
        x = x.view(x.size(0), -1)
        x = self.dense_activation_fn(self.fc1(x))
        return self.fc2(x)

# Activation Function Map
activation_map = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "mish": F.mish
}

# DataLoader Function
def get_dataloaders(data_dir, batch_size, val_split=0.2, augment=False):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ]) if augment else transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    full_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
    label_to_indices = {}
    for idx, (_, label) in enumerate(full_dataset.samples):
        label_to_indices.setdefault(label, []).append(idx)

    train_idx, val_idx = [], []
    for indices in label_to_indices.values():
        random.shuffle(indices)
        split = int(len(indices) * val_split)
        val_idx.extend(indices[:split])
        train_idx.extend(indices[split:])

    train_data = Subset(full_dataset, train_idx)
    val_data = Subset(ImageFolder(os.path.join(data_dir, 'train'), transform=transform_val), val_idx)

    return DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2), \
           DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2), \
           len(full_dataset.classes)

# Training Loop
def train_epoch(model, optimizer, criterion, dataloader, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(dataloader), 100. * correct / total

# Evaluation Loop
def evaluate(model, criterion, dataloader, device):
    model.eval()
    loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return loss / len(dataloader), 100. * correct / total

# CLI Entry Point
def main(args):
    wandb.login(key=args.wandb_key)
    wandb.init(project=args.project_name, config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filter_map = {
        'same': [args.base_filter]*5,
        'double': [args.base_filter*(2**i) for i in range(5)],
        'half': [max(1, args.base_filter//(2**i)) for i in range(5)]
    }

    train_loader, val_loader, num_classes = get_dataloaders(
        args.data_dir, args.batch_size, val_split=0.2, augment=args.augment)

    model = Cutomized_CNN(
        input_channels=3,
        conv_filters=filter_map[args.filter_organization],
        kernel_sizes=[3]*5,
        activation_fn=activation_map[args.activation_fn],
        dense_neurons=args.dense_neurons,
        dense_activation_fn=activation_map[args.activation_fn],
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        num_classes=num_classes
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader, device)
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss, "train_accuracy": train_acc,
            "val_loss": val_loss, "val_accuracy": val_acc
        })

    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

# ArgParser for CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Custom CNN with CLI")

    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to save the trained model')
    parser.add_argument('--wandb_key', type=str, required=True, help='Your wandb API key')
    parser.add_argument('--project_name', type=str, default="CNN-Training", help='WandB project name')
    parser.add_argument('--base_filter', type=int, default=32, help='Base filter size')
    parser.add_argument('--filter_organization', choices=['same', 'double', 'half'], default='double')
    parser.add_argument('--activation_fn', choices=['relu', 'gelu', 'silu', 'mish'], default='relu')
    parser.add_argument('--batch_norm', action='store_true', help='Use batch normalization')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--dense_neurons', type=int, default=128, help='Dense layer neurons')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')

    args = parser.parse_args()
    main(args)
