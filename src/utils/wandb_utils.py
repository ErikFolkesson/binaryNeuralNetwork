import wandb

from src.utils.path_utils import get_logs_dir

def setup_wandb(training_config, experiment_id=None):
    """Set up Weights & Biases logging."""
    logs_dir = get_logs_dir() / "simple_cnn"
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"experiment_{experiment_id}" if experiment_id is not None else None

    wandb.init(
        project="simple_cnn_project",
        name=run_name,
        config={
            "epochs": training_config.epochs,
            "batch_size": training_config.train_batch_size,
            "learning_rate": training_config.learning_rate
        },
        reinit=True,
        dir=str(logs_dir)
    )

def log_wandb(data: dict):
    wandb.log(data)

def watch_wandb(model):
    wandb.watch(model, log = "all")