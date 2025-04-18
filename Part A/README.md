# CNN Classifier with W&B Hyperparameter Tuning  
### Assignment Notebook: `MA23M026_A2_PartA.ipynb`

Hi This is my notebook where I’ve built a custom Convolutional Neural Network CNN using PyTorch. I also connected it with Weights and Biases W&B to do a Bayesian hyperparameter sweep. The whole thing is super modular and easy to experiment with.

This assignment is mainly designed to work with the iNaturalist 12K image classification dataset, but you can use any dataset with a similar folder structure.

The model reads images from the train folder and automatically splits it into training and validation sets while maintaining class balance.

## Code Breakdown

### CNN Model Cutomized_CNN class
This is where I built a flexible CNN using torch.nn.Module. You can control almost everything:

- Number of convolutional filters per layer
- Kernel sizes
- Activation function ReLU GELU SiLU Mish
- Dropout helps avoid overfitting
- Batch normalization optional
- Dense fully connected layer size
- Number of output classes

The model builds multiple convolutional layers dynamically using ModuleList, and then adds two fully connected layers at the end.

The forward pass is defined in forward_pass which  
1. Applies conv activation pooling layers.  
2. Flattens the result.  
3. Passes through dense layer and final output layer.

### Activation Function Mapper
There’s a dictionary called activation_map that maps string names like "gelu" or "mish" to actual PyTorch activation functions. This makes it super easy to plug them into the model dynamically.

You can pass activation names in the W&B config and it automatically applies them.

### Data Loading and Splitting get_dataloaders
This function:

- Loads image data using ImageFolder
- Applies image resizing and optional data augmentation
- Splits data into train and validation sets using stratified sampling keeps class balance
- Returns PyTorch DataLoaders for training and validation

If augmentation is on, it adds random flips and rotations to help improve generalization.

### Training and Evaluation
There are two helper functions:

- train_epoch runs one training loop over the data, computes loss and accuracy.
- evaluate runs one pass over validation data without gradient updates.

Both give easy-to-read outputs like train and validation loss and accuracy for each epoch.

### Training with W&B Sweeps train function
This is where the real magic happens. Inside the train function:

- W&B picks hyperparameters from the sweep config.
- It builds the CNN using the selected values.
- Trains and validates the model for N epochs.
- Logs everything loss accuracy learning rate etc. to your W&B dashboard.

Here’s what gets auto-tuned:

- Base number of filters 32 or 64
- Filter organization same double or half
- Activation function GELU SiLU Mish
- Whether to use BatchNorm
- Dropout rate
- Size of the dense FC layer
- Batch size
- Learning rate
- Number of epochs

### W&B Sweep Config
Here I’ve defined a Bayesian sweep config, which finds the best hyperparameter combination over time.

You just run it once, and W&B will take care of training multiple models with different settings and keep track of everything

### About Files

```
MA23M026_A2_PartA.ipynb
```

This Jupyter notebook contains the complete analysis and explanation for Assignment-2 (DA6401) Part A.

```
ma23m026_a2_parta.py
```

This Python script includes all the classes and functions related to the customized CNN architecture. You can import this file into any other script or notebook by keeping it in the same directory.

```
trainA.py
```

This is the main training script. You can train a sample CNN model using command-line arguments. See the "How to Run" section below for detailed instructions.

## How to Run trainA.py (CLI)

1. Make sure W&B is installed:
   
   ```
   pip install torch torchvision wandb matplotlib

   ```

2. Login to W&B once:
   
   ```
   wandb login "YOUR_WANDB_API_KEY"

   ```

3. Dataset Structure:

   ```
   inaturalist_12K/
   ├── train/
   │   ├── class_1/
   │   │   ├── img1.jpg
   │   │   └── ...
   │   ├── class_2/
   │   └── ...

   ```

4. Command-Line Usage

```
python trainA.py \
  --data_dir /path/to/inaturalist_12K
```
**OR**

```
python trainA.py \
  --data_dir /path/to/inaturalist_12K \
  --batch_size 64 \
  --lr 0.001 \
  --base_filter 32 \
  --filter_organization double \
  --activation_fn relu \
  --dense_neurons 128 \
  --dropout 0.3 \
  --batch_norm True \
  --data_augmentation True \
  --epochs 10
```

## 
Run the sweep (optional):

   ```
   sweep_id = wandb.sweep(sweep_config, project='iNaturalist-CNN-PartA-BayesianSearch')  
   wandb.agent(sweep_id, function=train, count=10)
   ```

This will run 10 training jobs with different hyperparameter values and log all metrics to W&B.


## Wandb Report
https://wandb.ai/viinod9-iitm/MA23M026_DA6401_A2/reports/MA23M026_DA6401_Assignment-2--VmlldzoxMjA5NzcwNg

## Github Repo Link
https://github.com/viinod9/DA6401_MA23M026_A2/tree/main/Part%20A


## Summary of Features

Feature | Description  
--------|-------------  
Custom CNN | Built from scratch, super configurable  
Stratified Data Split | Maintains label balance in train and validation  
Data Augmentation | Optional, adds robustness  
W&B Integration | Logs everything, plus hyperparameter search  
Multiple Activation Support | Use ReLU GELU SiLU Mish  
Modular Code | Easy to reuse and modify

## Extra Notes

- The image size is fixed to 224x224 – common for pre-trained models too.
- Model uses CrossEntropyLoss and Adam optimizer.
- Everything is modular – so you can tweak it for your own datasets easily.
- Great for learning how CNNs and W&B sweeps work together.

