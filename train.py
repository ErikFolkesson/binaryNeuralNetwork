from src.config import TrainingConfig
from src.utils.logger import setup_logger
from src.models.model import MyModel
import torch.optim as optim

if __name__ == "__main__":
    config = TrainingConfig()
    setup_logger(config.log_dir)

    model = MyModel()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Add your training loop
