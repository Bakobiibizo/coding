import os
import pytest
from src.vectordb.VectorDB import VectorDB
from loguru import logger

def test_vectordb_operations():
    # Get environment variables
    url = os.getenv("WEAVIATE_URL")
    api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not url or not api_key:
        pytest.skip("WEAVIATE_URL or WEAVIATE_API_KEY not set")
    
    # Initialize VectorDB
    db = VectorDB(url=url, api_key=api_key, use_local=False)
    
    # Test class creation
    class_name = "TestCollection"
    vectorizer_config = {}  # Will use default config from create_class
    
    try:
        # Create class
        logger.info("Creating test collection...")
        db.create_class(class_name, vectorizer_config)
        
        # Test adding objects
        logger.info("Testing object addition...")
        test_objects = [
            {
                "content": "This is a test document 1",
                "metadata": {
                    "filename": "test1.txt",
                    "language": "en",
                    "type": "test"
                }
            },
            {
                "content": "This is a test document 2",
                "metadata": {
                    "filename": "test2.txt",
                    "language": "en",
                    "type": "test"
                }
            }
        ]
        db.add_objects(class_name, test_objects)
        
        # Test querying objects
        logger.info("Testing object querying...")
        results = db.query_objects(
            class_name=class_name,
            query_text="test document",
            limit=2
        )
        logger.info(f"Query results: {results}")
        assert len(results) > 0, "No results found"
        assert "content" in results[0], "Result missing content"
        assert "metadata" in results[0], "Result missing metadata"
        
        # Test filtered query
        logger.info("Testing filtered query...")
        filtered_results = db.query_objects(
            class_name=class_name,
            query_text="test document",
            limit=1,
            filters={
                "path": ["metadata", "filename"],
                "operator": "Equal",
                "valueText": "test1.txt"
            }
        )
        logger.info(f"Filtered results: {filtered_results}")
        assert len(filtered_results) == 1, "Filtered query returned wrong number of results"
        assert filtered_results[0]["metadata"]["filename"] == "test1.txt", "Wrong document returned"
        
        # Test deleting specific objects
        logger.info("Testing object deletion with filter...")
        db.delete_objects(
            class_name=class_name,
            filter_dict={
                "path": ["metadata", "filename"],
                "operator": "Equal",
                "valueText": "test1.txt"
            }
        )
        
        # Verify deletion
        results_after_delete = db.query_objects(
            class_name=class_name,
            query_text="test document",
            limit=2
        )
        assert len(results_after_delete) == 1, "Object was not deleted properly"
        
        # Test deleting entire collection
        logger.info("Testing collection deletion...")
        db.delete_objects(class_name)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Cleanup - make sure to delete the test collection
        try:
            db.delete_objects(class_name)
        except:
            pass
        
if __name__ == "__main__":
    test_vectordb_operations()
