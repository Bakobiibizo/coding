from typing import Optional, Any
from loguru import logger
import traceback

class VectorDBError(Exception):
    """Base exception for VectorDB operations"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error
        logger.error(f"{message} - Original error: {original_error}")

class GeneratorError(Exception):
    """Base exception for text generator operations"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error
        logger.error(f"{message} - Original error: {original_error}")

class ChunkerError(Exception):
    """Base exception for chunking operations"""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error
        logger.error(f"{message} - Original error: {original_error}")

def handle_errors(func):
    """Decorator for consistent error handling"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise VectorDBError(error_msg, e)
    return wrapper

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging settings"""
    config = {
        "handlers": [
            {
                "sink": "stdout",
                "format": "{time} | {level} | {message}",
                "level": log_level,
            }
        ]
    }
    
    if log_file:
        config["handlers"].append({
            "sink": log_file,
            "format": "{time} | {level} | {message}",
            "level": log_level,
            "rotation": "500 MB",
        })
    
    logger.configure(**config)