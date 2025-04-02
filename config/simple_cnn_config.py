from dataclasses import dataclass

@dataclass
class TrainingConfig:
    train_batch_size: int = 32
    test_batch_size: int = 32
    eval_batch_size: int = 32
    eval_split = 0.2
    learning_rate: float = 1e-3
    epochs: int = 3