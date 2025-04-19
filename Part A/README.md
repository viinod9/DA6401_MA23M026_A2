# CNN Classifier with W&B Hyperparameter Tuning  
Project File: MA23M026_A2_PartA.ipynb

Hi This is my notebook where I’ve built a custom Convolutional Neural Network CNN using PyTorch. I also connected it with Weights and Biases W&B to do a Bayesian hyperparameter sweep. The whole thing is super modular and easy to experiment with.

This project is mainly designed to work with the iNaturalist 12K image classification dataset, but you can use any dataset with a similar folder structure.

## Dataset Format

Make sure your dataset directory looks like this:

/inaturalist_12K/  
train/  
class1/  
class2/  
...

The model reads images from the train folder and automatically splits it into training and validation sets while maintaining class balance.

## Code Breakdown

### 1. CNN Model Cutomized_CNN class
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

### 2. Activation Function Mapper
There’s a dictionary called activation_map that maps string names like "gelu" or "mish" to actual PyTorch activation functions. This makes it super easy to plug them into the model dynamically.

You can pass activation names in the W&B config and it automatically applies them.

### 3. Data Loading and Splitting get_dataloaders
This function:

- Loads image data using ImageFolder
- Applies image resizing and optional data augmentation
- Splits data into train and validation sets using stratified sampling keeps class balance
- Returns PyTorch DataLoaders for training and validation

If augmentation is on, it adds random flips and rotations to help improve generalization.

### 4. Training and Evaluation
There are two helper functions:

- train_epoch runs one training loop over the data, computes loss and accuracy.
- evaluate runs one pass over validation data without gradient updates.

Both give easy-to-read outputs like train and validation loss and accuracy for each epoch.

### 5. Training with W&B Sweeps train function
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

### 6. W&B Sweep Config
Here I’ve defined a Bayesian sweep config, which finds the best hyperparameter combination over time.

You just run it once, and W&B will take care of training multiple models with different settings and keep track of everything

## How to Run

1. Make sure W&B is installed:
   pip install wandb

2. Login to W&B once:
   wandb login

3. Place your dataset at:
   /kaggle/input/inaturalist-dataset/inaturalist_12K/train/

4. Run the sweep:
   sweep_id = wandb.sweep(sweep_config, project='iNaturalist-CNN-PartA-BayesianSearch')  
   wandb.agent(sweep_id, function=train, count=10)

This will run 10 training jobs with different hyperparameter values and log all metrics to W&B.

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

Let me know if you want this in PDF or markdown file or if you'd like help writing Part B of your assignment too
