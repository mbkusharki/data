import torch
import numpy as np
import random
import logging
import os

def set_seed(seed=42):
    """Ensure reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_logger(log_path="outputs/training.log"):
    """Set up a logger for training and FL progress tracking."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("FL_Logger")
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger
