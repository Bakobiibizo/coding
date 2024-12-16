import pytest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import MagicMock

class MockWeaviateClient:
    """Mock Weaviate client for testing"""
    def __init__(self):
        self.schema = MagicMock()
        self.batch = MagicMock()
        self.query = MagicMock()
        self.data_object = MagicMock()

class VectorTestUtils:
    @staticmethod
    def generate_random_vector(dim: int = 768) -> List[float]:
        """Generate random vector for testing"""
        return list(np.random.rand(dim))

    @staticmethod
    def create_test_chunk(
        text: str,
        doc_name: str,
        chunk_id: str,
        vector: Optional[List[float]] = None
    ):
        """Create a test chunk"""
        from src.vectordb.chunkers.chunk import Chunk
        chunk = Chunk(
            text=text,
            doc_name=doc_name,
            doc_type="test",
            chunk_id=chunk_id
        )
        if vector:
            chunk.set_vector(vector)
        return chunk

    @staticmethod
    def create_test_document(
        text: str,
        name: str,
        chunks: Optional[List[Any]] = None
    ):
        """Create a test document"""
        from src.vectordb.readers.document import Document
        doc = Document(
            text=text,
            name=name,
            doc_type="test",
            reader="test"
        )
        if chunks:
            doc.chunks = chunks
        return doc

@pytest.fixture
def mock_weaviate_client():
    """Fixture for mock Weaviate client"""
    return MockWeaviateClient()

@pytest.fixture
def vector_test_utils():
    """Fixture for vector test utilities"""
    return VectorTestUtils()

@pytest.fixture
def sample_chunks(vector_test_utils):
    """Fixture for sample chunks"""
    return [
        vector_test_utils.create_test_chunk(
            f"Test chunk {i}",
            "test_doc",
            str(i),
            vector_test_utils.generate_random_vector()
        )
        for i in range(5)
    ]