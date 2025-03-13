import unittest
import torch
import os
from pathlib import Path

from src.models.baseline_cnn import SimpleCNN, get_data_loaders, train_model
from src.utils.path_utils import get_models_dir

class TestBaselineCNN(unittest.TestCase):
    def test_model_structure(self):
        """Test that the model initializes with the expected structure."""
        model = SimpleCNN()

        # Check layers exist
        self.assertIsInstance(model.conv1, torch.nn.Conv2d)
        self.assertIsInstance(model.conv2, torch.nn.Conv2d)
        self.assertIsInstance(model.fc1, torch.nn.Linear)
        self.assertIsInstance(model.fc2, torch.nn.Linear)
        self.assertIsInstance(model.pool, torch.nn.MaxPool2d)

        # Check parameters
        self.assertEqual(model.conv1.in_channels, 1)
        self.assertEqual(model.conv1.out_channels, 32)
        self.assertEqual(model.conv2.in_channels, 32)
        self.assertEqual(model.conv2.out_channels, 64)
        self.assertEqual(model.fc1.in_features, 64 * 7 * 7)
        self.assertEqual(model.fc1.out_features, 128)
        self.assertEqual(model.fc2.in_features, 128)
        self.assertEqual(model.fc2.out_features, 10)

    def test_forward_pass(self):
        """Test that forward pass produces correct output shape."""
        model = SimpleCNN()
        batch_size = 4
        x = torch.randn(batch_size, 1, 28, 28)  # MNIST image shape

        # Test forward pass
        output = model(x)
        self.assertEqual(output.shape, (batch_size, 10))

    def test_get_data_loaders(self):
        """Test that data loaders are created correctly."""
        train_loader, test_loader = get_data_loaders()

        # Check that loaders exist
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(test_loader)

        # Get a batch to verify batch shape
        images, labels = next(iter(train_loader))

        # Check data types and shapes
        self.assertEqual(images.dim(), 4)  # [batch_size, channels, height, width]
        self.assertEqual(images.shape[1], 1)  # 1 channel for grayscale
        self.assertEqual(images.shape[2], 28)  # Height
        self.assertEqual(images.shape[3], 28)  # Width
        self.assertEqual(labels.dim(), 1)  # [batch_size]

    def test_training_step(self):
        """Test that a single training step works."""
        model = SimpleCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        # Create mock data (small batch)
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))

        # Forward pass
        output = model(x)
        loss = criterion(output, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that loss is a valid float
        self.assertIsInstance(loss.item(), float)

    def test_model_save(self):
        """Test that the model can be saved."""
        model = SimpleCNN()
        model_dir = get_models_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        test_path = model_dir / "test_model.pt"

        # Save the model
        torch.save(model.state_dict(), test_path)

        # Check that the file exists
        self.assertTrue(test_path.exists())

        # Clean up
        if test_path.exists():
            os.remove(test_path)

if __name__ == '__main__':
    unittest.main()