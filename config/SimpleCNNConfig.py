from dataclasses import dataclass

@dataclass
class TrainingConfig:
    train_batch_size: int = 32
    test_batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 3
    log_dir: str = "C:\\Users\\erik\\IES_codebase\\pythonProjects\\binaryNeuralNetwork\\logs\\SimpleCNN"