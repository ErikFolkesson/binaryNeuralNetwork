# Project Title

## Setup

## Experiment Tracking
This project uses Weights & Biases for experiment tracking. 

## Data
### MNIST
The MNIST dataset is used for training and evaluating the models. Downloaded using the `torchvision` library.

# TODO
- [ ] Add automatic hyperparameter search

## Done
- [X] Make sure multiple training runs in a row are correctly saved in wandb (currently the steps just keep counting)
- [X] Add a path util to make sure the paths are correct
- [X] Modify train.py to train multiple SimpleCNN's
- [X] Add evaluation metrics to SimpleCNN
- [X] Add final evaluation of SimpleCNN using test set