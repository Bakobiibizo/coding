# src/config.py
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from loguru import logger
import yaml

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from environment variables and optional config file.
    
    Args:
        config_path: Optional path to a YAML configuration file
        
    Returns:
        Dictionary containing configuration values
    """
    # Load environment variables
    load_dotenv()
    
    config = {
        # API Configuration
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "weaviate_url": os.getenv("WEAVIATE_URL"),
        "weaviate_api_key": os.getenv("WEAVIATE_API_KEY"),
        
        # Generator Configuration
        "model": os.getenv("AGENTARTIFICIAL_MODEL", "gpt-4"),
        "max_tokens": int(os.getenv("MAX_TOKENS", "1024")),
        "temperature": float(os.getenv("TEMPERATURE", "0.7")),
        
        # Database Configuration
        "vector_db": {
            "url": os.getenv("WEAVIATE_URL"),
            "api_key": os.getenv("WEAVIATE_API_KEY"),
            "batch_size": int(os.getenv("VECTOR_BATCH_SIZE", "100")),
            "distance_threshold": float(os.getenv("VECTOR_DISTANCE_THRESHOLD", "0.7"))
        }
    }
    
    # Load additional configuration from file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                config.update(file_config)
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            raise
    
    # Validate required configuration
    required_keys = ["openai_api_key", "weaviate_url", "weaviate_api_key"]
    missing_keys = [key for key in required_keys if not config.get(key)]
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")
    
    return config