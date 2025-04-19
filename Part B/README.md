# Fine-tuning pre-trained model (ResNet-50)

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
## 🚀 How to Run

### Step 1: Download the Dataset

Download the ```https://www.kaggle.com/datasets/viinod9/inaturalist-dataset```  and ensure it is placed at:

```
/kaggle/input/inaturalist-dataset/inaturalist_12K/train

```

### Step 2: Configure WandB
Replace the following line in the notebook:

wandb.login(key="YOUR_WANDB_API_KEY")
With your actual WandB API key 

### Step 3: Run the Notebook
Open the notebook and run all cells. It will:
Initialize a WandB sweep

### Sweep Search Space
The WandB sweep explores the following combinations:

```

epochs: 10, 15, 20

freeze_ratio: 0.8

architecture: resnet

learning_rate: 0.001, 0.0001, 0.01

augment: True, False

```

Each sweep run uses a unique combination of these to optimize validation accuracy.

