from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 10
    log_dir: str = "logs/"

    # W&B specific
    wandb_project: str = "binaryNeuralNetwork"
    wandb_entity: str = "erik-folkesson1-efml"
    wandb_tags: list = field(default_factory=lambda: ["baseline", "experiment"])
    log_dir: str = "logs/"