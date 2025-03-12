from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 10
    log_dir: str = "logs/"