"""
Training script for SimpleCNN model on MNIST dataset.

This script sets up and runs the training process for the SimpleCNN model
using the training functionality defined in the baseline_cnn module.
"""

import argparse
from src.models.baseline_cnn import train_model


def main(args):
    """
    Set up and execute multiple training runs with the specified parameters.

    Args:
        args: Command-line arguments containing training configuration
    """
    print(
        f"Starting SimpleCNN model training: {args.num_experiments} experiments with base name '{args.experiment_name}'")

    for i in range(args.num_experiments):
        print(f"\nRunning experiment {i + 1}/{args.num_experiments}")
        # Run the training process
        train_model(use_wandb=args.use_wandb, experiment_id=f"{args.experiment_name}_{i + 1}")

    print("All training runs completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimpleCNN on MNIST dataset")
    parser.add_argument("--experiment_name", type=str, default="simple_cnn_run",
                        help="Name for the training experiment")
    parser.add_argument("--num_experiments", type=int, default=1,
                        help="Number of experiments to run")
    parser.add_argument("--use_wandb", type=bool, default=False,
                        help="Use Weights and Biases during training")

    args = parser.parse_args()
    main(args)
