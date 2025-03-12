import unittest
from src.data.mnist_torchvision_data_loader import get_data

class TestMNISTDataLoader(unittest.TestCase):
    def test_get_data(self):
        train_data, test_data = get_data("../data/raw")
        self.assertEqual(len(train_data), 60000)
        self.assertEqual(len(test_data), 10000)


if __name__ == '__main__':
    unittest.main()

