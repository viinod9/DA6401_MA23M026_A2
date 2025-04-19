import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import os
import numpy as np
import random
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import models
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


wandb.login(key="YOUR_API_KEY")



# Defining a custom Convolutional Neural Network (CNN) class
class Cutomized_CNN(nn.Module):
    def __init__(
        self,
        input_channels=3,  # Number of input channels, e.g., 3 for RGB images
        conv_filters=[32, 64, 128, 256, 512],  # Number of filters for each conv layer
        kernel_sizes=[3, 3, 3, 3, 3],  # Kernel size for each conv layer
        activation_fn=F.relu,  # Activation function to be used after conv layers
        dense_neurons=256,  # Number of neurons in the fully connected hidden layer
        dense_activation_fn=F.relu,  # Activation function after dense layer
        dropout=0.0,  # Dropout rate (0 means no dropout)
        batch_norm=False,  # Whether to apply batch normalization after conv layers
        num_classes=10  # Number of output classes (for classification)
    ):
        # Initializing the parent nn.Module class
        super(Cutomized_CNN, self).__init__()

        # Saving configuration values
        self.activation_fn = activation_fn
        self.dense_activation_fn = dense_activation_fn
        self.dropout = dropout
        self.batch_norm = batch_norm

        # Initializing a list to hold all convolutional blocks
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels  # Start with input channels (e.g., 3 for RGB)

        # Creating convolutional layers one by one
        for out_channels, kernel_size in zip(conv_filters, kernel_sizes):
            layers = []

            # Add convolutional layer with given kernel size and padding
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))

            # Optionally add batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))

            # Add max pooling layer to reduce spatial dimensions
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            # Group the layers into a sequential block and add to the list
            self.conv_layers.append(nn.Sequential(*layers))

            # Update input channel count for the next conv layer
            in_channels = out_channels

        # Creating a dummy input tensor to compute the size after flattening
        self._dummy_input = torch.zeros(1, input_channels, 224, 224)

        # Compute the flattened size of the feature map after all conv layers
        self.flattened_size = self._get_flattennig_size()

        # Defining the first fully connected (dense) layer
        fc1_layers = []

        # Optionally add dropout before dense layer to reduce overfitting
        if dropout > 0:
            fc1_layers.append(nn.Dropout(dropout))

        # Add a linear layer from flattened input to dense_neurons
        fc1_layers.append(nn.Linear(self.flattened_size, dense_neurons))
        self.fc1 = nn.Sequential(*fc1_layers)

        # Final output layer from dense_neurons to number of classes
        self.fc2 = nn.Linear(dense_neurons, num_classes)

    # Helper function to calculate the flattened size after conv layers
    def _get_flattennig_size(self):
        x = self._dummy_input

        # Pass the dummy input through all conv blocks
        for block in self.conv_layers:
            for layer in block:
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
                    x = layer(x)
                elif isinstance(layer, nn.MaxPool2d):
                    x = layer(x)

        # Flatten the output and return the number of features
        return x.view(1, -1).size(1)

    # forward pass of the network
    def forward_pass(self, x):
        # Pass the input through each conv block
        for block in self.conv_layers:
            for layer in block:
                if isinstance(layer, nn.Conv2d):
                    # Apply convolution and then activation
                    x = layer(x)
                    x = self.activation_fn(x)
                else:
                    # Apply either batch norm or max pooling
                    x = layer(x)

        # Flatten the output before feeding into dense layers
        x = x.view(x.size(0), -1)

        # Pass through first dense layer
        x = self.fc1(x)

        # Apply activation function
        x = self.dense_activation_fn(x)

        # Pass through final output layer
        x = self.fc2(x)

        # Return the final output (logits)
        return x

# """# Model Checking"""


# model = Cutomized_CNN(
#     input_channels=3,
#     conv_filters=[16, 32, 64, 128, 256],
#     kernel_sizes=[3, 3, 3, 3, 3],
#     activation_fn=F.relu,
#     dense_neurons=128,
#     dense_activation_fn=F.relu,
#     dropout=0.3,
#     batch_norm=True,
#     num_classes=10
# )

# # Print the architecture of the model to verify its structure
# print(model)

# activation_map = {
#     "relu": nn.ReLU(),
#     "gelu": nn.GELU(),
#     "silu": nn.SiLU(),
#     "mish": nn.Mish()
# }

# Returns the chosen activation function wrapped in a lambda for usage.
def get_activation_function(name):
    return lambda x: activation_map[name.lower()](x)


def get_dataloaders(data_dir, batch_size, val_split=0.2, augment=False):
    # Define transformation pipeline for training data
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),   # Data augmentation - flips image
        transforms.RandomRotation(10),       # Data augmentation - rotates image
        transforms.ToTensor()
    ]) if augment else transforms.Compose([
        transforms.Resize((224, 224)),       # Only resizing and converting to tensor
        transforms.ToTensor()
    ])

 
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load the full dataset (from 'train' folder) using training transform
    full_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)

    # Group indices of images by their class labels for stratified splitting
    label_to_indices = {}
    for idx, (_, label) in enumerate(full_dataset.samples):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    train_idx = []
    val_idx = []

    # Stratified split: maintain class distribution in both train and val sets
    for label in label_to_indices:
        indices = label_to_indices[label]
        random.shuffle(indices)
        split = int(len(indices) * val_split)
        val_idx.extend(indices[:split])
        train_idx.extend(indices[split:])

    # Create Subset objects for training and validation
    train_data = Subset(full_dataset, train_idx)
    val_data = Subset(ImageFolder(os.path.join(data_dir, 'train'), transform=transform_val), val_idx)

    # Wrap the subsets in DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # Return loaders and number of classes
    return train_loader, val_loader, len(full_dataset.classes)


def train_epoch(model, optimizer, criterion, dataloader, device):
    model.train()  # Set model to training mode
    running_loss, correct, total = 0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()             # Clear gradients
        outputs = model(images)           # forward pass
        loss = criterion(outputs, labels) # Compute loss
        loss.backward()                   # Backward pass
        optimizer.step()                  # Update model parameters

        running_loss += loss.item()
        _, predicted = outputs.max(1)     # Get class predictions
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()  # Count correct predictions

    # Return average loss and accuracy for the epoch
    return running_loss / len(dataloader), 100. * correct / total


# Evaluation function (no gradient calculation)
def evaluate(model, criterion, dataloader, device):
    model.eval()  # Set model to evaluation mode
    loss, correct, total = 0, 0, 0

    with torch.no_grad():  # No gradient updates during evaluation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Return average validation loss and accuracy
    return loss / len(dataloader), 100. * correct / total


def train(config=None):
    with wandb.init(config=config):  # Start a new wandb run
        config = wandb.config  # Get sweep config

        # 🔽 Generate a unique name for each run based on config
        run_name = (
            f"filt-{config.base_filter}_{config.filter_organization}_"
            f"act-{config.activation_fn}_bn-{config.batch_norm}_"
            f"do-{config.dropout}_dense-{config.dense_neurons}_"
            f"bs-{config.batch_size}_lr-{config.lr}_aug-{config.data_augmentation}"
        )
        wandb.run.name = run_name

        # Set the device to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load data loaders with optional augmentation
        train_loader, val_loader, num_classes = get_dataloaders(
            data_dir='/kaggle/input/inaturalist-dataset/inaturalist_12K',
            batch_size=config.batch_size,
            val_split=0.2,
            augment=config.data_augmentation
        )

        # Select filter sizes based on the selected strategy: same, double, or half
        conv_filters = {
            'same': [config.base_filter]*5,
            'double': [config.base_filter*(2**i) for i in range(5)],
            'half': [config.base_filter//(2**i) for i in range(5)],
        }[config.filter_organization]

        # Instantiate the CNN model with selected hyperparameters
        model = Cutomized_CNN(
            conv_filters=conv_filters,
            kernel_sizes=[3]*5,
            activation_fn=get_activation_function(config.activation_fn),
            dense_neurons=config.dense_neurons,
            dense_activation_fn=F.relu,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
            num_classes=num_classes
        )

        model.to(device)  # Move model to GPU/CPU

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        # Run training for specified number of epochs
        for epoch in range(config.epochs):
            train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader, device)
            val_loss, val_acc = evaluate(model, criterion, val_loader, device)

            # Print progress like Keras-style format
            print(f"Epoch : {epoch + 1}/{config.epochs}")
            print(f" -- Train Accuracy: {train_acc:.4f} -- Validation Accuracy: {val_acc:.4f}")

            # Log metrics to wandb for visualization
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

# """# Bayes Search with 10 runs"""

# # ------------- Sweep Config

# sweep_config = {
#     'method': 'bayes',
#     'metric': {'name': 'val_acc', 'goal': 'maximize'},
#     'parameters': {
#         'base_filter': {'values': [32, 64]},
#         'filter_organization': {'values': ['same', 'double']},
#         'activation_fn': {'values': ['gelu', 'silu', 'mish']},
#         'data_augmentation': {'values': [True]},
#         'batch_norm': {'values': [True, False]},
#         'dropout': {'values': [0.2, 0.3]},
#         'dense_neurons': {'values': [128, 256]},
#         'batch_size': {'values': [32, 64, 128]},
#         'lr': {'values': [1e-3, 1e-4]},
#         'epochs': {'values': [10, 15, 20]}
#     }
# }

# sweep_id = wandb.sweep(sweep_config, project='iNaturalist-CNN-PartA-BayesianSearch')
# wandb.agent(sweep_id, function=train, count = 10)



# # Define the transformation for test images: resize and convert to tensor
# transform_test = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize images to 224x224
#     transforms.ToTensor()           # Convert PIL images to PyTorch tensors
# ])

# # Load the test dataset from the specified directory with the defined transformations
# test_dataset = ImageFolder('/kaggle/input/inaturalist-dataset/inaturalist_12K/val', transform=transform_test)

# # Create a DataLoader for the test dataset
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)  # No shuffle for evaluation

# # Get the class names from the dataset (used for predictions or visualization)
# class_names = test_dataset.classes

def plot_test_predictions(model, dataloader, class_names, device, num_images=30):
    model.eval()  # Set the model to evaluation mode
    images_shown = 0
    fig, axes = plt.subplots(10, 3, figsize=(12, 30))  # Create a 10x3 grid for displaying images
    axes = axes.flatten()  # Flatten the grid for easy indexing

    with torch.no_grad():  # Disable gradient calculation for inference
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)  # Move data to device
            outputs = model(images)  # Get model predictions
            _, preds = torch.max(outputs, 1)  # Get predicted class index

            for i in range(images.size(0)):
                img = images[i].cpu().permute(1, 2, 0).numpy()  # Convert image tensor to numpy for plotting
                label = class_names[labels[i]]  # Get ground truth class name
                pred = class_names[preds[i]]    # Get predicted class name

                axes[images_shown].imshow(img)  # Show image
                axes[images_shown].set_title(f"GT: {label}\nPred: {pred}")  # Annotate with GT and prediction
                axes[images_shown].axis('off')  # Turn off axis display

                images_shown += 1
                if images_shown == num_images:
                    plt.tight_layout()  # Adjust layout

                    # 🔽 Log figure to wandb
                    wandb.log({"Test Predictions": wandb.Image(fig)})

                    plt.close(fig)  # Close plot to free memory
                    return  # Exit after logging desired number of images

def evaluate_test(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")



# activation_map = {
#     "relu": nn.ReLU(),
#     "gelu": nn.GELU(),
#     "silu": nn.SiLU(),
#     "mish": nn.Mish()
# }

# # Returns the chosen activation function wrapped in a lambda for usage.
def get_activation_function(name):
    return lambda x: activation_map[name.lower()](x)


def get_dataloaders(data_dir, batch_size, val_split=0.2, augment=False):
    # Define transformation pipeline for training data
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),   # Data augmentation - flips image
        transforms.RandomRotation(10),       # Data augmentation - rotates image
        transforms.ToTensor()
    ]) if augment else transforms.Compose([
        transforms.Resize((224, 224)),       # Only resizing and converting to tensor
        transforms.ToTensor()
    ])

    # Define transformation pipeline for validation data
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load the full dataset (from 'train' folder) using training transform
    full_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)

    # Group indices of images by their class labels for stratified splitting
    label_to_indices = {}
    for idx, (_, label) in enumerate(full_dataset.samples):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    train_idx = []
    val_idx = []

    # Stratified split: maintain class distribution in both train and val sets
    for label in label_to_indices:
        indices = label_to_indices[label]
        random.shuffle(indices)
        split = int(len(indices) * val_split)
        val_idx.extend(indices[:split])
        train_idx.extend(indices[split:])

    # Create Subset objects for training and validation
    train_data = Subset(full_dataset, train_idx)
    val_data = Subset(ImageFolder(os.path.join(data_dir, 'train'), transform=transform_val), val_idx)

    # Wrap the subsets in DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # Return loaders and number of classes
    return train_loader, val_loader, len(full_dataset.classes)


# ------------- Training and Evaluation Functions -------------------

# One training pass through the dataset (single epoch)
def train_epoch(model, optimizer, criterion, dataloader, device):
    model.train()  # Set model to training mode
    running_loss, correct, total = 0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()             # Clear gradients
        outputs = model(images)           # forward pass
        loss = criterion(outputs, labels) # Compute loss
        loss.backward()                   # Backward pass
        optimizer.step()                  # Update model parameters

        running_loss += loss.item()
        _, predicted = outputs.max(1)     # Get class predictions
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()  # Count correct predictions

    # Return average loss and accuracy for the epoch
    return running_loss / len(dataloader), 100. * correct / total


# Evaluation function (no gradient calculation)
def evaluate(model, criterion, dataloader, device):
    model.eval()  # Set model to evaluation mode
    loss, correct, total = 0, 0, 0

    with torch.no_grad():  # No gradient updates during evaluation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Return average validation loss and accuracy
    return loss / len(dataloader), 100. * correct / total


# ------------- Train Loop for wandb Sweep -------------------
# Full training loop for use with wandb hyperparameter sweeps
def trainForTest(config=None):
    with wandb.init(config=config):  # Start a new wandb run
        config = wandb.config  # Get sweep config

        # 🔽 Generate a unique name for each run based on config
        run_name = (
            f"filt-{config.base_filter}_{config.filter_organization}_"
            f"act-{config.activation_fn}_bn-{config.batch_norm}_"
            f"do-{config.dropout}_dense-{config.dense_neurons}_"
            f"bs-{config.batch_size}_lr-{config.lr}_aug-{config.data_augmentation}"
        )
        wandb.run.name = run_name

        # Set the device to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load data loaders with optional augmentation
        train_loader, val_loader, num_classes = get_dataloaders(
            data_dir='/kaggle/input/inaturalist-dataset/inaturalist_12K',
            batch_size=config.batch_size,
            val_split=0.2,
            augment=config.data_augmentation
        )

        # Select filter sizes based on the selected strategy: same, double, or half
        conv_filters = {
            'same': [config.base_filter]*5,
            'double': [config.base_filter*(2**i) for i in range(5)],
            'half': [config.base_filter//(2**i) for i in range(5)],
        }[config.filter_organization]

        # Instantiate the CNN model with selected hyperparameters
        model = Cutomized_CNN(
            conv_filters=conv_filters,
            kernel_sizes=[3]*5,
            activation_fn=get_activation_function(config.activation_fn),
            dense_neurons=config.dense_neurons,
            dense_activation_fn=F.relu,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
            num_classes=num_classes
        )

        model.to(device)  # Move model to GPU/CPU

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        # Run training for specified number of epochs
        for epoch in range(config.epochs):
            train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader, device)
            val_loss, val_acc = evaluate(model, criterion, val_loader, device)


            # Print progress like Keras-style format
            print(f"Epoch : {epoch + 1}/{config.epochs}")
            print(f" -- Train Accuracy: {train_acc:.4f} -- Validation Accuracy: {val_acc:.4f}")


            # Log metrics to wandb for visualization
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

        evaluate_test(model, test_loader, device)
        plot_test_predictions(model, test_loader, class_names, device)

# sweep_config = {
#     'method': 'random',
#     'metric': {'name': 'val_acc', 'goal': 'maximize'},
#     'parameters': {
#         'base_filter': {'values': [32]},
#         'filter_organization': {'values': ['double']},
#         'activation_fn': {'values': ['mish']},
#         'data_augmentation': {'values': [True]},
#         'batch_norm': {'values': [True]},
#         'dropout': {'values': [0.2]},
#         'dense_neurons': {'values': [256]},
#         'batch_size': {'values': [64]},
#         'lr': {'values': [1e-4]},
#         'epochs': {'values': [5]}
#     }
# }

# sweep_id = wandb.sweep(sweep_config, project='iNaturalist-CNN-testnew1')
# wandb.agent(sweep_id, function=trainForTest, count = 1)


def visualize_first_layer_filters(model):
    # Extract filters from the first convolutional layer
    filters = model.conv_layers[0][0].weight.data.clone()

    # Normalize filters to range [0, 1] for visualization
    filters = (filters - filters.min()) / (filters.max() - filters.min())

    # Create a grid of filter images
    grid = make_grid(filters, nrow=8, normalize=True, padding=1)

    # Create a new figure for plotting
    plt.figure(figsize=(8, 8))

    # Set title for the plot
    plt.title("Conv1 Filters")

    # Display the filters (convert CHW to HWC for matplotlib)
    plt.imshow(grid.permute(1, 2, 0).cpu())

    # Remove axis ticks
    plt.axis('off')

    # Show the plot
    plt.show()

# Visualize the filters of the best model
visualize_first_layer_filters(best_model)


class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None

        # Hook to modify gradient flow through ReLU (only allow positive gradients)
        def relu_hook_function(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0.0),)

        # Register backward hook to all ReLU activations
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(relu_hook_function)

    def generate_gradients(self, input_image, target_layer_idx=4, filter_idx=0):
        # Enable gradient tracking for the input image
        input_image.requires_grad = True

        # forward pass up to the target convolutional layer
        x = input_image
        for i in range(target_layer_idx + 1):
            x = self.model.conv_layers[i](x)

        # Backward pass to compute gradients w.r.t. the input image
        x[:, filter_idx].mean().backward(retain_graph=True)

        # Process gradients for visualization
        grad = input_image.grad[0].cpu().permute(1, 2, 0).numpy()
        grad = (grad - grad.min()) / (grad.max() - grad.min())
        return grad

def plot_guided_backprop(model, dataloader, num_filters=10):
    # Create GuidedBackprop object
    gbp = GuidedBackprop(model)

    # Take the first image from the dataloader
    for images, _ in dataloader:
        image = images[0:1].to(device)
        break

    # Plot gradients for each filter in the specified layer
    fig, axes = plt.subplots(num_filters, 1, figsize=(5, num_filters * 3))
    for i in range(num_filters):
        grad = gbp.generate_gradients(image.clone(), filter_idx=i)
        axes[i].imshow(grad)
        axes[i].axis('off')
        axes[i].set_title(f"Neuron {i} in CONV5")

    # Adjust subplot spacing
    plt.tight_layout()
    plt.show()

# # Visualize guided backpropagation for selected neurons
# plot_guided_backprop(best_model, test_loader)

