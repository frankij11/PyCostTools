"""Logging configuration utilities for the pycost package."""

import logging
import os
import sys

def setup_logging(log_level=logging.INFO, log_file=None):
    """Configure logging for the module.
    
    Args:
        log_level: The logging level (default: INFO)
        log_file: Optional file path for logging (default: None - console only)
    
    Returns:
        Logger instance for this module
    """
    logger = logging.getLogger('pycost.analysis')
    logger.setLevel(log_level)
    logger.propagate = False
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create formatters and handlers
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger 