import pytest
import os
from typing import List
from src.vectordb.VectorDB import VectorDB 
from src.text_generators.AgentArtificialGenerator import AgentArtificialGenerator

@pytest.mark.integration
class TestVectorDBIntegration:
    @pytest.fixture
    def vector_db(self):
        return VectorDB(
            url=os.getenv("TEST_WEAVIATE_URL"),
            api_key=os.getenv("TEST_WEAVIATE_API_KEY")
        )

    @pytest.fixture
    def generator(self):
        return AgentArtificialGenerator()

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(
        self,
        vector_db,
        generator,
        sample_chunks,
        vector_test_utils
    ):
        # Test class creation
        class_name = "TestClass"
        vector_config = {
            "vectorizer": "text2vec-openai",
            "vector_index_config": {
                "distance": "cosine"
            }
        }
        vector_db.create_class(class_name, vector_config)

        # Add vectors
        vector_db.add_vectors(class_name, sample_chunks)

        # Generate embeddings for search
        query = "Test search query"
        response = await generator.generate([query], [])
        
        # Search vectors
        results = vector_db.search_vectors(
            class_name,
            vector_test_utils.generate_random_vector(),
            limit=2
        )
        
        assert len(results) <= 2
        assert all("content" in result for result in results)

        # Update vector
        if results:
            vector_db.update_vector(
                class_name,
                results[0]["id"],
                {"content": "Updated content"},
                vector_test_utils.generate_random_vector()
            )

        # Delete vectors
        vector_db.delete_vectors(
            class_name,
            {"path": ["content"], "operator": "Equal", "valueText": "Updated content"}
        )

@pytest.mark.integration
class TestGeneratorIntegration:
    @pytest.fixture
    def generator(self):
        return AgentArtificialGenerator()

    @pytest.mark.asyncio
    async def test_generation_workflow(self, generator):
        # Test basic generation
        query = "Write a test query"
        response = await generator.generate([query], [])
        assert response and isinstance(response, str)

        # Test streaming
        async for chunk in generator.generate_stream([query], []):
            assert isinstance(chunk, dict)
            assert "message" in chunk
            assert "finish_reason" in chunk