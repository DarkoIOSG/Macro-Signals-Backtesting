import logging
import sys
from pathlib import Path


def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Create a logger with consistent formatting.
    
    Parameters:
    -----------
    name : str, logger name (usually __name__)
    log_level : str, logging level
    
    Returns:
    --------
    logging.Logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, log_level.upper()))
        
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger