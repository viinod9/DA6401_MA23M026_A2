# 📘 ma23m026_a2_partb.ipynb

## Fine-tuning pre-trained model (ResNet-50)

This notebook (`ma23m026_a2_partb.ipynb`) implements a **transfer learning** pipeline for fine-tuning pretrained CNN models (ResNet50 and InceptionV3) on the iNaturalist 12K dataset. It includes:

- Configurable architectures and training hyperparameters
- Layer freezing strategies
- Optional data augmentation
- Real-time logging with **Weights & Biases (WandB)**
- Automated hyperparameter tuning via **Bayesian sweeps**

---

## Key Features

- Pretrained model fine-tuning (ResNet50, InceptionV3)
- Configurable layer freezing (`freeze_ratio`, `last_only`)
- Optional image augmentation
- Logs training/validation metrics to **WandB**
- Hyperparameter tuning using **Bayesian optimization sweeps**

---

## Functions Overview

- `get_model(name, freeze_ratio, last_only)`: Loads the model, modifies the final layer, and applies freezing logic.
- `prepare_data(data_path, augment)`: Applies transforms and splits data into training and validation sets.
- `run_training(model, train_dl, val_dl, num_epochs, cfg)`: Trains the model and logs results.
- `sweep_runner()`: Executes one training run with parameters sampled from WandB sweep.
- `sweep_configuration`: Defines the hyperparameter sweep search space.

---

## Requirements

Make sure you have the following Python packages installed:

```
pip install torch torchvision scikit-learn matplotlib wandb

```
