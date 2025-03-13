import unittest
from pathlib import Path
from src.utils.path_utils import (
    get_project_root,
    get_raw_data_dir,
    get_processed_data_dir,
    get_logs_dir,
    get_models_dir
)

class TestPathUtils(unittest.TestCase):

    def test_get_project_root(self):
        """Test that get_project_root returns a valid Path object."""
        root_path = get_project_root()
        self.assertIsInstance(root_path, Path)
        self.assertTrue(root_path.exists())

        # Check that essential directories exist in the project root
        self.assertTrue((root_path / "src").exists())

    def test_get_raw_data_dir(self):
        """Test that get_raw_data_dir returns the correct raw data directory."""
        data_dir = get_raw_data_dir()
        self.assertIsInstance(data_dir, Path)

        # Verify it has the correct structure
        self.assertEqual(data_dir.name, "raw")
        self.assertEqual(data_dir.parent.name, "data")

        # Should be project_root/data/raw
        self.assertEqual(data_dir, get_project_root() / "data" / "raw")

    def test_get_processed_data_dir(self):
        """Test that get_processed_data_dir returns the correct processed data directory."""
        data_dir = get_processed_data_dir()
        self.assertIsInstance(data_dir, Path)

        # Verify it has the correct structure
        self.assertEqual(data_dir.name, "processed")
        self.assertEqual(data_dir.parent.name, "data")

        # Should be project_root/data/processed
        self.assertEqual(data_dir, get_project_root() / "data" / "processed")

    def test_get_logs_dir(self):
        """Test that get_logs_dir returns the correct logs directory."""
        logs_dir = get_logs_dir()
        self.assertIsInstance(logs_dir, Path)

        # Should be project_root/logs
        self.assertEqual(logs_dir, get_project_root() / "logs")

    def test_get_models_dir(self):
        """Test that get_models_dir returns the correct models directory."""
        models_dir = get_models_dir()
        self.assertIsInstance(models_dir, Path)

        # Should be project_root/models
        self.assertEqual(models_dir, get_project_root() / "models")


if __name__ == '__main__':
    unittest.main()