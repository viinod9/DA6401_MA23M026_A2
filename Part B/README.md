# Fine-tuning pre-trained model (ResNet-50)
### Assignment Notebook : `ma23m026_a2_partb.ipynb`
Hi, I implemented a **transfer learning** pipeline for fine-tuning pretrained CNN models (ResNet50 and InceptionV3) on the iNaturalist 12K dataset. It includes:

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

```bash
pip install torch torchvision scikit-learn matplotlib wandb
```

---

### About Files

```
ma23m026_a2_partb.ipynb
```

This Jupyter notebook contains the complete analysis and explanation for Assignment-2 (DA6401) Part B.

```
ma23m026_a2_partb.py
```

This Python script includes all the classes and functions related to the Fine-tuning pre-trained model (ResNet-50) with further hyperparameter tuning. You can import this file into any other script or notebook by keeping it in the same directory.

```
trainB.ipynb
```

This is the main training script. You can train a sample Fine-tuning pre-trained model (ResNet-50) using command-line arguments. See the "How to Run" section below for detailed instructions.

---


## How to Run `trainB.py` (CLI)

1. Make sure W&B is installed:
   
   ```
   pip install torch torchvision scikit-learn matplotlib wandb

   ```

2. Login to W&B once:
   
   ```
   wandb login "YOUR_WANDB_API_KEY"

   ```

3. Dataset Structure:
   Prepare your image dataset in the following format (standard for torchvision.datasets.ImageFolder)

   ```
   inaturalist_12K/
   ├── train/
   │   ├── class_1/
   │   │   ├── img1.jpg
   │   │   └── ...
   │   ├── class_2/
   │   └── ...

   ```

5. Command-Line Usage

```
python trainB.py \
  --data_dir /path/to/inaturalist_12K
```
**OR**
```

python trainB.py \
  --data_path /path/to/your/dataset \
  --wandb_key your_wandb_api_key \
  --architecture resnet \
  --freeze_ratio 0.8 \
  --epochs 10 \
  --learning_rate 0.001 \
  --augment

```



## How to Run `ma23m026_a2_partb.ipynb`

### Step 1: Download the Dataset

Download the [iNaturalist 12K dataset](https://www.kaggle.com/datasets/viinod9/inaturalist-dataset) and ensure it is placed at:

```bash
/kaggle/input/inaturalist-dataset/inaturalist_12K/train
```

### Step 2: Configure WandB

Replace the following line in the notebook:

```python
wandb.login(key="YOUR_WANDB_API_KEY")
```

With your actual WandB API key.

### Step 3: Run the Notebook

Open the notebook and run all cells. It will:

- Initialize a WandB sweep
- Run 10 experiments with varying hyperparameters
- Log metrics and save a plot for each run

---

## Sweep Search Space

The WandB sweep explores the following combinations:

- `epochs`: 10, 15, 20  
- `freeze_ratio`: 0.8  
- `architecture`: resnet  
- `learning_rate`: 0.001, 0.0001, 0.01  
- `augment`: True, False

Each sweep run uses a unique combination of these to optimize validation accuracy.

---

## Wandb Report
https://wandb.ai/viinod9-iitm/MA23M026_DA6401_A2/reports/MA23M026_DA6401_Assignment-2--VmlldzoxMjA5NzcwNg

## Github Repo Link
https://github.com/viinod9/DA6401_MA23M026_A2/tree/main/Part%20B

---

## Notes

- Assumes 10 output classes. Modify `model.fc = nn.Linear(..., 10)` if needed.
- All images resized to 224×224
- Only `resnet` architecture is used in sweep config, but Inception is supported manually

---


