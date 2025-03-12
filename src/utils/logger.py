import logging
from pathlib import Path

def setup_logger(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir/'training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )