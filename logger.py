import logging
import os

def setup_logger(name='segformer', log_dir='logs', filename='training.log'):
    """Set up a logger with both file and console output.

    Args:
        name (str): Name of the logger instance.
        log_dir (str): Directory to save log files.
        filename (str): Name of the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers if they don't exist already
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger