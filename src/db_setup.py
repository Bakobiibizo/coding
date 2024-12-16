# src/db_setup.py
from typing import Dict, Any
from loguru import logger
from src.vectordb.VectorDB import VectorDB

async def setup_database(config: Dict[str, Any]) -> VectorDB:
    """
    Set up and initialize the vector database connection.
    
    Args:
        config: Configuration dictionary containing database settings
        
    Returns:
        Initialized VectorDB instance
    """
    try:
        logger.info("Initializing vector database connection...")
        
        # Initialize VectorDB with configuration
        vector_db = VectorDB(
            url=config["vector_db"]["url"],
            api_key=config["vector_db"]["api_key"]
        )
        
        # Define vectorizer configuration
        vectorizer_config = {
            "vectorizer": "text2vec-openai",  # Using OpenAI's text vectorizer
            "vector_index_config": {
                "distance": "cosine",
                "ef": 100,  # Size of the dynamic list for the nearest neighbors
                "maxConnections": 64,  # Maximum number of connections per element
                "efConstruction": 128,  # Size of the dynamic list during construction
            },
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The actual content of the code or documentation",
                    "vectorizer": "text2vec-openai",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
                },
                {
                    "name": "metadata",
                    "dataType": ["object"],
                    "description": "Additional metadata about the content",
                    "properties": [
                        {
                            "name": "filename",
                            "dataType": ["text"],
                            "description": "Name of the file containing the content"
                        },
                        {
                            "name": "language",
                            "dataType": ["text"],
                            "description": "Programming language of the content"
                        },
                        {
                            "name": "type",
                            "dataType": ["text"],
                            "description": "Type of content (code or documentation)"
                        }
                    ]
                }
            ]
        }
        
        # Create or ensure existence of necessary classes
        class_names = ["CodeSnippets", "Documentation"]
        for class_name in class_names:
            try:
                vector_db.create_class(class_name, vectorizer_config)
                logger.info(f"Created vector class: {class_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"Vector class already exists: {class_name}")
                else:
                    raise
        
        logger.info("Vector database initialization complete")
        return vector_db
        
    except Exception as e:
        logger.error(f"Failed to initialize vector database: {e}")
        raise