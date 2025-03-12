import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

def get_data(data_dir):
    train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=ToTensor())
    return train_data, test_data