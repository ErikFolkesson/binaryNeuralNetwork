import unittest
import torch
from pathlib import Path
from src.data.mnist_torchvision_data_loader import get_data, get_train_eval_test_loaders
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

    def test_train_eval_test_loaders_default(self):
        # Test with default parameters
        train_loader, eval_loader, test_loader = get_train_eval_test_loaders()

        # Check batch size
        self.assertEqual(train_loader.batch_size, 32)
        self.assertEqual(eval_loader.batch_size, 32)
        self.assertEqual(test_loader.batch_size, 32)

        # Check dataset sizes (with default 20% eval split)
        self.assertEqual(len(train_loader.dataset), 48000)  # 80% of 60000
        self.assertEqual(len(eval_loader.dataset), 12000)  # 20% of 60000
        self.assertEqual(len(test_loader.dataset), 10000)

    def test_train_eval_test_loaders_custom_batch_sizes(self):
        # Test with custom batch sizes
        train_loader, eval_loader, test_loader = get_train_eval_test_loaders(
            train_batch_size=64, eval_batch_size=128, test_batch_size=256
        )

        self.assertEqual(train_loader.batch_size, 64)
        self.assertEqual(eval_loader.batch_size, 128)
        self.assertEqual(test_loader.batch_size, 256)

    def test_train_eval_test_loaders_custom_eval_split(self):
        # Test with custom eval split of 30%
        train_loader, eval_loader, test_loader = get_train_eval_test_loaders(eval_split=0.3)

        self.assertEqual(len(train_loader.dataset), 42000)  # 70% of 60000
        self.assertEqual(len(eval_loader.dataset), 18000)  # 30% of 60000
        self.assertEqual(len(test_loader.dataset), 10000)

    def test_train_eval_test_loaders_custom_path(self):
        # Test with custom path
        custom_path = get_raw_data_dir()
        train_loader, eval_loader, test_loader = get_train_eval_test_loaders(data_dir=custom_path)

        self.assertEqual(len(train_loader.dataset) + len(eval_loader.dataset), 60000)
        self.assertEqual(len(test_loader.dataset), 10000)

    def test_train_eval_test_loaders_seed_reproducibility(self):
        # Test that the same seed gives the same split
        seed = 123
        train_loader1, eval_loader1, _ = get_train_eval_test_loaders(random_seed=seed)
        train_loader2, eval_loader2, _ = get_train_eval_test_loaders(random_seed=seed)

        # Check that datasets have the same indices
        self.assertEqual(train_loader1.dataset.indices, train_loader2.dataset.indices)
        self.assertEqual(eval_loader1.dataset.indices, eval_loader2.dataset.indices)

        # Different seeds should give different splits
        train_loader3, eval_loader3, _ = get_train_eval_test_loaders(random_seed=seed + 1)
        self.assertNotEqual(train_loader1.dataset.indices, train_loader3.dataset.indices)


if __name__ == '__main__':
    unittest.main()
