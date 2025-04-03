"""
Baseline CNN model implementation for MNIST digit classification.

This module provides a simple CNN architecture for MNIST classification,
along with utilities for data loading and model training. The implementation
integrates with Weights & Biases for experiment tracking.
"""
from pathlib import Path
from typing import Union, Any

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


from src.utils.wandb_utils import setup_wandb, log_wandb, watch_wandb
from src.utils.model_utils import save_best_model
from src.utils.logging_util import setup_logging, log_metrics
from src.data.mnist_torchvision_data_loader import get_train_eval_test_loaders

import config.simple_cnn_config as TrainingConfig
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

def initialize_model():
    """Initialize model, loss function, and optimizer."""
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TrainingConfig.learning_rate)
    return model, criterion, optimizer

def train_epoch(model, criterion, optimizer, train_loader, epoch, epochs, use_wandb, print_interval=100):
    """Train the model for a single epoch."""
    model.train()
    running_loss = 0.0
    total_running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass and optimization
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Logging
        running_loss += loss.item()
        total_running_loss += loss.item()
        if batch_idx % print_interval == print_interval-1:
            avg_loss = running_loss / print_interval
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {avg_loss:.4f}')
            if use_wandb:
                log_wandb({"epoch": epoch + 1, "step": batch_idx + 1, "train_loss": avg_loss})
            running_loss = 0.0

    return total_running_loss / len(train_loader)

def evaluate_model(model, criterion, data_loader, phase="val"):
    """Evaluate the model on the given data loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in data_loader:
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(data_loader)

    metrics = {
        f"{phase}_loss": avg_loss,
        f"{phase}_accuracy": accuracy
    }

    return accuracy, avg_loss, metrics

def train_model(use_wandb = False, logging = True, experiment_id=None):
    """Train the SimpleCNN model on MNIST dataset with experiment tracking."""
    # Data loading
    train_loader, eval_loader, test_loader = get_train_eval_test_loaders(
        train_batch_size=TrainingConfig.train_batch_size,
        eval_batch_size=TrainingConfig.eval_batch_size,
        test_batch_size=TrainingConfig.test_batch_size,
        eval_split=TrainingConfig.eval_split
    )

    # Model initialization
    model, criterion, optimizer = initialize_model()

    if use_wandb:
        setup_wandb(TrainingConfig, experiment_id)
        watch_wandb(model)


    train_loss_list = []
    val_accuracy_list = []
    val_loss_list = []
    step_list = []

    if logging:
        logs_dir = setup_logging("simple_cnn")

    # Training
    best_accuracy = 0.0
    for epoch in range(TrainingConfig.epochs):
        # Train
        train_loss = train_epoch(model, criterion, optimizer, train_loader, epoch, TrainingConfig.epochs, use_wandb=use_wandb, print_interval=500)

        # Evaluate
        val_accuracy, val_loss, val_metrics = evaluate_model(model, criterion, eval_loader, "val")
        if use_wandb:
            log_wandb({"epoch": epoch + 1, **val_metrics})

        print(f'Epoch [{epoch + 1}/{TrainingConfig.epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')

        if logging:
            step_list.append((epoch + 1) * len(train_loader))
            train_loss_list.append(train_loss)
            val_accuracy_list.append(val_accuracy)
            val_loss_list.append(val_loss)

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_best_model(model, "simple_cnn", experiment_id)

    print("Training finished.")

    if logging:
        logging_dict = {
            "step": step_list,
            "train_loss": train_loss_list,
            "val_accuracy": val_accuracy_list,
            "val_loss": val_loss_list
        }
        # logs_dir will be initialized, always.
        log_metrics("simple_cnn", logging_dict)

    # Test evaluation
    test_accuracy, test_loss, test_metrics = evaluate_model(model, criterion, test_loader, "test")

    if use_wandb:
        log_wandb(test_metrics)

    print(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

    return model

if __name__ == "__main__":
    train_model()