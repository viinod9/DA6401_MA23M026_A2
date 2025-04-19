CNN Classifier with W and B Hyperparameter Tuning  
Project File MA23M026_A2_PartA.ipynb

This project builds a custom Convolutional Neural Network CNN using PyTorch and integrates it with Weights and Biases W and B for automatic hyperparameter tuning using Bayesian search. The code is tested with the iNaturalist 12K dataset but it can be used with any image dataset having a similar folder structure.

Dataset Format

Make sure your dataset directory is structured like this

your_dataset_directory
  train
    class1
    class2
    class3
    ...

The images should be inside class folders under the train directory. The code will automatically split the data into training and validation sets.

Code Explanation

1. CNN Model - Cutomized_CNN class

This is a flexible CNN model built using torch.nn.Module. It lets you customize

- Number of filters in convolutional layers
- Kernel sizes
- Activation function relu gelu silu mish
- Dropout value
- Use of batch normalization
- Dense layer size
- Number of output classes

It builds five convolutional blocks dynamically followed by a fully connected layer and an output layer.

2. Activation Function Mapper

The code includes a dictionary activation_map that lets you use activation function names as strings. The function get_activation_function returns the actual function for use in the model.

3. Data Loading and Splitting - get_dataloaders function

This function loads images using torchvision.datasets.ImageFolder. It performs stratified splitting of the data to ensure balanced class distribution. If augmentation is enabled it adds random horizontal flips and slight rotations to the training images. It returns DataLoaders for both train and validation sets.

4. Training and Evaluation

train_epoch runs one epoch of training and calculates the average loss and accuracy.

evaluate runs evaluation on the validation set and returns the average loss and accuracy. It does not calculate gradients.

5. W and B Sweep Training - train function

This function is used for W and B sweeps. It builds the CNN using parameters from the current W and B config and trains the model for a number of epochs. It logs training and validation accuracy and loss to the W and B dashboard.

The model can dynamically adjust based on the sweep parameters like

- base_filter number of filters
- filter_organization same double half
- activation_fn relu gelu silu mish
- data_augmentation true or false
- batch_norm true or false
- dropout rate
- dense_neurons number of dense units
- batch_size
- learning rate
- number of epochs

6. W and B Sweep Config

A sweep_config dictionary is defined that uses Bayesian optimization to maximize validation accuracy. W and B tries different combinations of parameters and selects the best.

How to Run

1. Install dependencies
pip install torch torchvision matplotlib wandb

2. Login to W and B once
wandb login

3. Make sure your dataset is available at
kaggle input inaturalist dataset inaturalist_12K train

4. Run the sweep from the notebook
sweep_id = wandb.sweep(sweep_config, project='iNaturalist-CNN-PartA-BayesianSearch')
wandb.agent(sweep_id, function=train, count=10)

This will run 10 experiments with different parameter combinations automatically and log everything to your W and B dashboard.

Summary of Features

Custom CNN with configurable layers and activation functions  
Stratified train and validation split  
Optional data augmentation  
Full W and B integration with sweep support  
Modular code with reusable components  
CrossEntropyLoss and Adam optimizer used  

Extra Notes

All images are resized to 224 by 224  
You can easily modify the dataset path or hyperparameters  
Code supports GPU acceleration if available  
Each training run is tracked with a unique name using the hyperparameters  

This code is useful for experimenting with CNNs and learning how to use W and B for managing experiments and hyperparameter tuning

