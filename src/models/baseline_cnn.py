"""
Baseline CNN model implementation for MNIST digit classification.

This module provides a simple CNN architecture for MNIST classification,
along with utilities for data loading and model training. The implementation
integrates with Weights & Biases for experiment tracking.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

import wandb

import src.data.mnist_torchvision_data_loader as mnist
import config.SimpleCNNConfig as TrainingConfig
TrainingConfig = TrainingConfig.TrainingConfig

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for MNIST digit classification.

    Architecture:
        - 2 convolutional layers with max pooling
        - 2 fully connected layers

    Input: [batch_size, 1, 28, 28] MNIST images
    Output: [batch_size, 10] class probabilities
    """
    def __init__(self):
        """Initialize the CNN architecture with convolutional and fully connected layers."""
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, 28, 28]

        Returns:
            torch.Tensor: Output logits of shape [batch_size, 10]
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_data_loaders():
    """
    Create and return PyTorch data loaders for MNIST dataset.

    Returns:
        tuple: (train_loader, test_loader) containing the data loaders
    """
    train_data, test_data = mnist.get_data()
    train_loader = mnist.get_data_loaders(train_data, TrainingConfig.train_batch_size)
    test_loader = mnist.get_data_loaders(test_data, TrainingConfig.test_batch_size)

    return train_loader, test_loader

def train_model(experiment_id=None):
    """
    Train the SimpleCNN model on MNIST dataset with experiment tracking.

    This function:
    1. Initializes Weights & Biases logging
    2. Sets up model, optimizer and loss function
    3. Trains the model for the configured number of epochs
    4. Saves the trained model weights
    """
    # Initialize W&B for experiment tracking
    run_name = f"experiment_{experiment_id}" if experiment_id is not None else None

    wandb.init(project="simple_cnn_project", name=run_name, config={
        "epochs": TrainingConfig.epochs,
        "batch_size": TrainingConfig.train_batch_size,
        "learning_rate": TrainingConfig.learning_rate
    }, reinit=True)

    train_loader, _ = get_data_loaders()
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TrainingConfig.learning_rate)

    wandb.watch(model, log="all")

    for epoch in range(TrainingConfig.epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass and optimization
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Logging
            running_loss += loss.item()
            if batch_idx % 100 == 99:  # Print every 100 batches
                avg_loss = running_loss / 100
                print(f'Epoch [{epoch + 1}/{TrainingConfig.epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                wandb.log({"epoch": epoch + 1, "step": batch_idx + 1, "loss": avg_loss})
                running_loss = 0.0

    print("Training finished.")

# Main execution
if __name__ == "__main__":
    train_model()