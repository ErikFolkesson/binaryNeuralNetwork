# Project Title

## Setup

## Experiment Tracking
This project uses Weights & Biases for experiment tracking. 

## Data
### MNIST
The MNIST dataset is used for training and evaluating the models. Downloaded using the `torchvision` library.

# TODO
- [ ] Add a path util to make sure the paths are correct
- [ ] Make sure multiple training runs in a row are correctly saved in wandb (currently the steps just keep counting)
- [ ] Add evaluation metrics to SimpleCNN
- [ ] Modify train.py to train multiple SimpleCNN's
- [ ] Add final evaluation using test set