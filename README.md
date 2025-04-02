# Project Title

## Setup

## Experiment Tracking
This project uses Weights & Biases for experiment tracking. 

## Data
### MNIST
The MNIST dataset is used for training and evaluating the models. Downloaded using the `torchvision` library.

## TODO
### TODO: Binary Neural Network: 
- [ ] 
### TODO: General Quality
- [ ] Create local logging
- [ ] Create visualization
  - [ ] Easy to reuse notebook 
  - [ ] Python file with helper function
- [ ] Setup linter and other code quality tools
- [ ] Add automatic hyperparameter search

### Done
- [X] Refractor and move CNN code to seperate files
- [X] Make it easier to turn off BandW. 
- [X] Make sure multiple training runs in a row are correctly saved in wandb (currently the steps just keep counting)
- [X] Add a path util to make sure the paths are correct
- [X] Modify train.py to train multiple SimpleCNN's
- [X] Add evaluation metrics to SimpleCNN
- [X] Add final evaluation of SimpleCNN using test set