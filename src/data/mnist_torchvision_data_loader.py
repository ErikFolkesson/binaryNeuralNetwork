import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from src.utils.path_utils import get_raw_data_dir

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