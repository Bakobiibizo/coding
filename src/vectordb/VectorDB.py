import weaviate
from weaviate.util import generate_uuid5
from weaviate.collections import Collection
from weaviate.collections.classes.config import (
    Configure, Property, DataType, VectorDistances,
    StopwordsPreset, Vectorizers
)
from typing import List, Dict, Optional, Any, Sequence
from loguru import logger
import numpy as np
from .chunkers.chunk import Chunk
import urllib.parse
import weaviate.classes as wvc

class VectorDB:
    """Interface to Weaviate vector database."""
    
    def __init__(self, url: str, api_key: str, use_local: bool = False):
        """Initialize Weaviate client.
        
        Args:
            url: Weaviate server URL
            api_key: API key for authentication
            use_local: If True, use local connection parameters, otherwise use cloud parameters
        """
        if use_local:
            # Parse the URL to get host and port for local instance
            parsed_url = urllib.parse.urlparse(url)
            host = parsed_url.hostname or "localhost"
            port = parsed_url.port or 8080
            
            # Configure connection parameters for local instance
            connection_params = weaviate.connect.ConnectionParams.from_params(
                http_host=host,
                http_port=port,
                http_secure=False,
                grpc_host=host,
                grpc_port=50051,  # Default gRPC port
                grpc_secure=False
            )
            
            # Initialize client with appropriate connection parameters
            self.client = weaviate.WeaviateClient(
                connection_params=connection_params,
                additional_headers={
                    "X-OpenAI-Api-Key": api_key
                }
            )
        else:
            # Configure connection parameters for cloud instance
            parsed_url = urllib.parse.urlparse(url)
            connection_params = weaviate.connect.ConnectionParams.from_params(
                http_host=parsed_url.netloc,
                http_port=443,
                http_secure=True,
                grpc_host=parsed_url.netloc,
                grpc_port=50051,  # Use a different port for gRPC
                grpc_secure=True
            )
            
            # Initialize client with appropriate connection parameters
            self.client = weaviate.WeaviateClient(
                connection_params=connection_params,
                skip_init_checks=True,  # Skip startup checks
                additional_config=wvc.init.AdditionalConfig(
                    timeout=wvc.init.Timeout(init=10)  # Increase init timeout
                ),
                auth_client_secret=wvc.init.Auth.api_key(api_key),
                additional_headers={
                    "X-OpenAI-Api-Key": api_key
                }
            )
        
        self.client.connect()
        
    def create_class(self, class_name: str, vectorizer_config: Dict[str, Any]) -> None:
        """Create a new class in Weaviate.
        
        Args:
            class_name: Name of the class to create
            vectorizer_config: Configuration for the vectorizer
        """
        try:
            collection = self.client.collections.create(
                name=class_name,
                properties=[
                    Property(
                        name="content",
                        data_type=DataType.TEXT,
                        description="The main content of the object",
                        vectorizer="text2vec-openai",
                        skip_vectorization=False,
                        vectorize_property_name=False,
                    ),
                    Property(
                        name="metadata",
                        data_type=DataType.OBJECT,
                        description="Additional metadata about the object",
                        nested_properties=[
                            Property(
                                name="filename",
                                data_type=DataType.TEXT,
                                description="Name of the file containing the content",
                            ),
                            Property(
                                name="language",
                                data_type=DataType.TEXT,
                                description="Programming language of the content",
                            ),
                            Property(
                                name="type",
                                data_type=DataType.TEXT,
                                description="Type of content (code or documentation)",
                            ),
                        ],
                    )
                ],
                inverted_index_config={
                    "bm25": {
                        "b": 0.75,
                        "k1": 1.2
                    },
                    "cleanupIntervalSeconds": 60,
                    "stopwords": {
                        "preset": StopwordsPreset.EN,
                        "additions": [],
                        "removals": []
                    },
                    "indexTimestamps": False,
                    "indexPropertyLength": False,
                    "indexNullState": False
                },
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                    ef=100,
                    max_connections=64,
                    ef_construction=128
                ),
                vectorizer_config=Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-small"
                )
            )
            
            return collection
            
        except Exception as e:
            logger.error(f"Failed to create class {class_name}: {str(e)}")
            raise
    
    def add_objects(
        self,
        class_name: str,
        objects: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> None:
        """Add objects to Weaviate in batches.
        
        Args:
            class_name: Name of the class to add objects to
            objects: List of objects to add
            batch_size: Number of objects to add in each batch
        """
        try:
            collection = self.client.collections.get(class_name)
            with collection.batch.dynamic() as batch:
                for obj in objects:
                    properties = {
                        "content": obj["content"],
                        "metadata": obj.get("metadata", {})
                    }
                    
                    # Use deterministic UUIDs based on content
                    uuid = generate_uuid5(obj["content"])
                    
                    batch.add_object(
                        properties=properties,
                        uuid=uuid
                    )
            logger.info(f"Added {len(objects)} objects to class: {class_name}")
        except Exception as e:
            logger.error(f"Error adding objects: {e}")
            raise
    
    def query_objects(
        self,
        class_name: str,
        query_text: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query objects from Weaviate using semantic search.
        
        Args:
            class_name: Name of the class to query
            query_text: Text to search for
            limit: Maximum number of results to return
            filters: Additional filters to apply
            
        Returns:
            List of matching objects
        """
        try:
            collection = self.client.collections.get(class_name)
            query = collection.query.near_text(
                query=query_text,
                limit=limit
            )
            
            if filters:
                query = query.with_where(filters)
            
            results = query.objects
            return [
                {
                    "content": obj.properties["content"],
                    "metadata": obj.properties.get("metadata", {}),
                    "_additional": {
                        "distance": obj.metadata.distance,
                        "id": obj.metadata.uuid
                    }
                }
                for obj in results
            ]
            
        except Exception as e:
            logger.error(f"Error querying objects: {e}")
            raise
    
    def delete_objects(
        self,
        class_name: str,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """Delete objects from Weaviate.
        
        Args:
            class_name: Name of the class to delete objects from
            filter_dict: Filter to select objects to delete
        """
        try:
            collection = self.client.collections.get(class_name)
            if filter_dict:
                # Delete objects matching the filter
                collection.data.delete_many(
                    where=filter_dict
                )
                logger.info(f"Deleted objects matching filter from class: {class_name}")
            else:
                # Delete the entire collection
                self.client.collections.delete(class_name)
                logger.info(f"Deleted entire class: {class_name}")
        except Exception as e:
            logger.error(f"Error deleting objects: {e}")
            raise