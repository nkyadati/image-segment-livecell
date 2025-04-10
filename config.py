import os
from pathlib import Path
import logging
import torch

# =========================
# Global Configuration
# =========================

class Config:
    """
    Centralized configuration class for SegFormer LIVECell segmentation.
    """

    # Project Paths
    try:
        # If running in a script, __file__ is defined
        DATA_ROOT = Path(__file__).parent.resolve()
    except NameError:
        # If running in a notebook or interactive shell, fallback to cwd
        DATA_ROOT = Path.cwd()
        
    MODEL_SAVE_PATH = os.path.join(DATA_ROOT, "trained_model")
    CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, "checkpoint.pth")

    # Image & Mask Directories
    TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "dataset/images/train")
    TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "dataset/annotations/semantic_masks/train")

    VAL_IMG_DIR = os.path.join(DATA_ROOT, "dataset/images/val")
    VAL_MASK_DIR = os.path.join(DATA_ROOT, "dataset/annotations/semantic_masks/val")

    TEST_IMG_DIR = os.path.join(DATA_ROOT, "dataset/images/test")
    TEST_MASK_DIR = os.path.join(DATA_ROOT, "dataset/annotations/semantic_masks/test")

    # Model / Training Settings
    NUM_CLASSES = 9
    BATCH_SIZE = 4
    LR = 5e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 50
    PATIENCE = 3
    
    MODEL_TYPE = "nvidia/mit-b3" # or "nvidia/mit-b5" for larger model or "nvidia/mit-b0" for smaller model
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    LOG_DIR = os.path.join(DATA_ROOT, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "training.log")

# Create config instance
cfg = Config()

# =========================
# Logger Configuration
# =========================

def setup_logger():
    """
    Sets up a global logger to track training and evaluation.
    """
    logger = logging.getLogger("SegFormer")
    logger.setLevel(logging.INFO)

    # Avoid duplicate log handlers
    if not logger.handlers:
        fh = logging.FileHandler(cfg.LOG_FILE)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

# Global logger
logger = setup_logger()
