import unittest
from pathlib import Path
from src.data.mnist_torchvision_data_loader import get_data
from src.utils.path_utils import get_raw_data_dir

class TestMNISTDataLoader(unittest.TestCase):
    def test_get_data_default_path(self):
        # Test using default path (from path_utils)
        train_data, test_data = get_data()
        self.assertEqual(len(train_data), 60000)
        self.assertEqual(len(test_data), 10000)

    def test_get_data_custom_path(self):
        # Test with custom path
        custom_path = get_raw_data_dir()  # Using path_utils to get a valid path
        train_data, test_data = get_data(custom_path)
        self.assertEqual(len(train_data), 60000)
        self.assertEqual(len(test_data), 10000)

if __name__ == '__main__':
    unittest.main()