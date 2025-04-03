import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from src.utils.path_utils import get_raw_data_dir
from torch.utils.data import random_split
import torch


def get_data(data_dir=None):
    """
    Get MNIST training and test datasets.

    Args:
        data_dir: Optional override for data directory path

    Returns:
        tuple: (train_data, test_data) containing the MNIST datasets
    """
    if data_dir is None:
        data_dir = get_raw_data_dir()

    train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=ToTensor())
    return train_data, test_data


def get_data_loaders(data, batch_size):
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data_loader


def get_train_eval_test_loaders(train_batch_size=32, eval_batch_size=32, test_batch_size=32,
                                eval_split=0.2, random_seed=42, data_dir=None):
    """
    Get MNIST train, evaluation, and test data loaders.

    Args:
        train_batch_size (int): Batch size for training data
        eval_batch_size (int): Batch size for evaluation data
        test_batch_size (int): Batch size for test data
        eval_split (float): Proportion of training data to use for evaluation (0.0 to 1.0)
        random_seed (int): Random seed for reproducible data splits
        data_dir: Optional override for data directory path

    Returns:
        tuple: (train_loader, eval_loader, test_loader) containing the data loaders
    """
    # Get the training and test datasets
    train_data, test_data = get_data(data_dir)

    # Calculate the split sizes
    train_size = int((1 - eval_split) * len(train_data))
    eval_size = len(train_data) - train_size

    # Split the training dataset
    generator = torch.Generator().manual_seed(random_seed)
    train_subset, eval_subset = random_split(train_data, [train_size, eval_size], generator=generator)

    # Create data loaders
    train_loader = get_data_loaders(train_subset, train_batch_size)
    eval_loader = get_data_loaders(eval_subset, eval_batch_size)
    test_loader = get_data_loaders(test_data, test_batch_size)

    return train_loader, eval_loader, test_loader
