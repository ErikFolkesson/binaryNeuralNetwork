import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def get_data(data_dir):
    train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=ToTensor())
    return train_data, test_data

def get_data_loaders(data, batch_size):
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data_loader