"""
Test file for semantic search blocks.
Tests the semantic search functionality and integration with existing blocks.
"""

import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from backend.blocks.semantic_search import (
    SemanticSearchBlock,
    EmbeddingGeneratorBlock,
    SemanticSimilarityBlock,
    HybridSearchBlock,
    embed_text,
    calculate_cosine_similarity,
    hybrid_search
)
from backend.blocks.search import EnhancedWikipediaSearchBlock
from backend.blocks.llm import SemanticLLMBlock
from backend.data.block import BlockSchemaInput, BlockSchemaOutput
from backend.data.model import APIKeyCredentials
from backend.util.settings import Secrets


class TestSemanticSearchBlock:
    """Test the semantic search block functionality."""
    
    @pytest.fixture
    def mock_credentials(self):
        """Create mock API credentials."""
        return APIKeyCredentials(
            id="test-cred",
            provider="openai",
            api_key="sk-test-key",
            title="Test Credentials"
        )
    
    @pytest.fixture
    def semantic_search_block(self):
        """Create a semantic search block instance."""
        return SemanticSearchBlock()
    
    @pytest.mark.asyncio
    async def test_semantic_search_basic(self, semantic_search_block, mock_credentials):
        """Test basic semantic search functionality."""
        # Create test input
        test_input = semantic_search_block.Input(
            query="machine learning algorithms",
            search_corpus=[
                "Machine learning is a subset of artificial intelligence",
                "Deep learning uses neural networks with multiple layers",
                "Supervised learning requires labeled training data",
                "Unsupervised learning finds patterns in unlabeled data"
            ],
            max_results=3,
            similarity_threshold=0.7
        )
        
        # Mock the embedding generation
        with patch('backend.blocks.semantic_search.embed_text') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            # Run the block
            results = []
            async for output in semantic_search_block.run(
                test_input,
                credentials=mock_credentials
            ):
                results.append(output)
            
            # Verify results
            assert len(results) > 0
            assert "results" in results[0]
            assert "query" in results[0]
            assert len(results[0]["results"]) <= 3
    
    @pytest.mark.asyncio
    async def test_semantic_search_with_threshold(self, semantic_search_block, mock_credentials):
        """Test semantic search with similarity threshold."""
        test_input = semantic_search_block.Input(
            query="artificial intelligence",
            search_corpus=[
                "AI is transforming various industries",
                "Natural language processing is a part of AI",
                "Cooking recipes from around the world"
            ],
            max_results=5,
            similarity_threshold=0.8  # High threshold
        )
        
        with patch('backend.blocks.semantic_search.embed_text') as mock_embed:
            # Mock embeddings with varying similarities
            mock_embed.side_effect = [
                [0.9, 0.1, 0.1, 0.1, 0.1],  # Query embedding
                [0.8, 0.2, 0.1, 0.1, 0.1],  # High similarity
                [0.7, 0.3, 0.1, 0.1, 0.1],  # Medium similarity
                [0.1, 0.1, 0.1, 0.1, 0.9]   # Low similarity
            ]
            
            results = []
            async for output in semantic_search_block.run(
                test_input,
                credentials=mock_credentials
            ):
                results.append(output)
            
            # Should only return results above threshold
            assert all(result["similarity_score"] >= 0.8 
                      for result in results[0]["results"])


class TestEmbeddingGeneratorBlock:
    """Test the embedding generator block."""
    
    @pytest.fixture
    def embedding_block(self):
        """Create an embedding generator block."""
        return EmbeddingGeneratorBlock()
    
    @pytest.fixture
    def mock_credentials(self):
        """Create mock credentials."""
        return APIKeyCredentials(
            id="test-cred",
            provider="openai",
            api_key="sk-test-key",
            title="Test Credentials"
        )
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_single_text(self, embedding_block, mock_credentials):
        """Test generating embeddings for a single text."""
        test_input = embedding_block.Input(
            texts=["Hello, world!"],
            model="text-embedding-3-small"
        )
        
        with patch('openai.AsyncOpenAI') as mock_client:
            # Mock the OpenAI response
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_client.return_value.embeddings.create = AsyncMock(return_value=mock_response)
            
            results = []
            async for output in embedding_block.run(
                test_input,
                credentials=mock_credentials
            ):
                results.append(output)
            
            assert "embeddings" in results[0]
            assert len(results[0]["embeddings"]) == 1
            assert len(results[0]["embeddings"][0]) == 3
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, embedding_block, mock_credentials):
        """Test generating embeddings for multiple texts."""
        test_input = embedding_block.Input(
            texts=["Text 1", "Text 2", "Text 3"],
            model="text-embedding-3-small",
            batch_size=2
        )
        
        with patch('openai.AsyncOpenAI') as mock_client:
            # Mock batch responses
            mock_response1 = MagicMock()
            mock_response1.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3]),
                MagicMock(embedding=[0.4, 0.5, 0.6])
            ]
            
            mock_response2 = MagicMock()
            mock_response2.data = [MagicMock(embedding=[0.7, 0.8, 0.9])]
            
            mock_client_instance = mock_client.return_value
            mock_client_instance.embeddings.create = AsyncMock(
                side_effect=[mock_response1, mock_response2]
            )
            
            results = []
            async for output in embedding_block.run(
                test_input,
                credentials=mock_credentials
            ):
                results.append(output)
            
            assert "embeddings" in results[0]
            assert len(results[0]["embeddings"]) == 3


class TestEnhancedWikipediaSearchBlock:
    """Test the enhanced Wikipedia search block with semantic capabilities."""
    
    @pytest.fixture
    def enhanced_search_block(self):
        """Create an enhanced Wikipedia search block."""
        return EnhancedWikipediaSearchBlock()
    
    @pytest.fixture
    def mock_credentials(self):
        """Create mock credentials."""
        return APIKeyCredentials(
            id="test-wiki-cred",
            provider="wikipedia",
            api_key="",
            title="Wikipedia Credentials"
        )
    
    @pytest.mark.asyncio
    async def test_enhanced_wikipedia_search(self, enhanced_search_block, mock_credentials):
        """Test enhanced Wikipedia search with semantic related topics."""
        test_input = enhanced_search_block.Input(
            topic="Artificial Intelligence",
            include_semantic_related=True,
            max_related_topics=3
        )
        
        # Mock both Wikipedia API and semantic search
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock Wikipedia summary response
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "extract": "Artificial intelligence (AI) is intelligence demonstrated by machines..."
            })
            mock_get.return_value.__aenter__.return_value = mock_response
            
            results = []
            async for output in enhanced_search_block.run(
                test_input,
                credentials=mock_credentials
            ):
                results.append(output)
            
            # Verify both summary and related topics are present
            assert "summary" in results[0]
            assert "related_topics" in results[0] or "related_topics" in results[1]
            assert "AI" in results[0]["summary"]


class TestSemanticLLMBlock:
    """Test the semantic LLM block."""
    
    @pytest.fixture
    def semantic_llm_block(self):
        """Create a semantic LLM block."""
        return SemanticLLMBlock()
    
    @pytest.fixture
    def mock_credentials(self):
        """Create mock OpenAI credentials."""
        return APIKeyCredentials(
            id="test-openai-cred",
            provider="openai",
            api_key="sk-test-key",
            title="OpenAI Credentials"
        )
    
    @pytest.mark.asyncio
    async def test_semantic_llm_with_context(self, semantic_llm_block, mock_credentials):
        """Test semantic LLM with context retrieval."""
        test_input = semantic_llm_block.Input(
            prompt="What did we learn about Python?",
            context_search="Python programming discussion",
            use_semantic_context=True,
            max_context_items=2,
            model="gpt-4o-mini"
        )
        
        # Mock OpenAI response
        with patch('backend.blocks.llm.get_openai_client') as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(
                message=MagicMock(content="Based on our discussion, Python is a versatile programming language...")
            )]
            mock_response.usage = MagicMock(
                prompt_tokens=50,
                completion_tokens=100,
                total_tokens=150
            )
            mock_client.return_value.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            
            results = []
            async for output in semantic_llm_block.run(
                test_input,
                credentials=mock_credentials
            ):
                results.append(output)
            
            # Verify response and context
            assert "response" in results[0]
            assert "retrieved_context" in results[1]
            assert "usage" in results[2]
            assert len(results[1]["retrieved_context"]) <= 2


class TestHybridSearch:
    """Test hybrid search functionality."""
    
    def test_hybrid_search_combines_results(self):
        """Test that hybrid search combines semantic and lexical results."""
        semantic_results = [
            {"content": "Deep learning with neural networks", "score": 0.9},
            {"content": "Machine learning algorithms", "score": 0.8}
        ]
        
        lexical_results = [
            {"content": "Neural network architectures", "score": 0.85},
            {"content": "Algorithm optimization", "score": 0.75}
        ]
        
        # Test hybrid search
        combined = hybrid_search(
            semantic_results=semantic_results,
            lexical_results=lexical_results,
            semantic_weight=0.6,
            lexical_weight=0.4
        )
        
        # Verify results are combined and re-ranked
        assert len(combined) == 4
        # Results should be sorted by combined score
        assert combined[0]["combined_score"] >= combined[1]["combined_score"]
    
    def test_calculate_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vector_a = [1, 0, 0]
        vector_b = [0, 1, 0]
        vector_c = [1, 0, 0]
        
        # Orthogonal vectors should have 0 similarity
        assert abs(calculate_cosine_similarity(vector_a, vector_b)) < 0.001
        
        # Identical vectors should have 1 similarity
        assert abs(calculate_cosine_similarity(vector_a, vector_c) - 1.0) < 0.001


class TestIntegration:
    """Integration tests for semantic search components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_semantic_search_workflow(self):
        """Test a complete semantic search workflow."""
        # Step 1: Generate embeddings for corpus
        corpus = [
            "Python is a high-level programming language",
            "JavaScript is used for web development",
            "Machine learning requires Python expertise",
            "Web frameworks include Django and React"
        ]
        
        with patch('openai.AsyncOpenAI') as mock_client:
            # Mock embedding generation
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3]),
                MagicMock(embedding=[0.2, 0.3, 0.4]),
                MagicMock(embedding=[0.3, 0.4, 0.5]),
                MagicMock(embedding=[0.4, 0.5, 0.6])
            ]
            mock_client.return_value.embeddings.create = AsyncMock(return_value=mock_response)
            
            # Generate embeddings
            embeddings = []
            for text in corpus:
                embedding = await embed_text(text, "text-embedding-3-small")
                embeddings.append(embedding)
            
            assert len(embeddings) == len(corpus)
            assert all(len(emb) == 3 for emb in embeddings)
            
            # Step 2: Perform semantic search
            query_embedding = await embed_text("programming languages", "text-embedding-3-small")
            
            similarities = []
            for i, corpus_embedding in enumerate(embeddings):
                similarity = calculate_cosine_similarity(query_embedding, corpus_embedding)
                similarities.append({
                    "text": corpus[i],
                    "similarity": similarity
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Verify results
            assert similarities[0]["similarity"] >= similarities[-1]["similarity"]
            assert all(0 <= s["similarity"] <= 1 for s in similarities)


# Performance tests
class TestPerformance:
    """Performance tests for semantic search."""
    
    @pytest.mark.asyncio
    async def test_large_corpus_search_performance(self):
        """Test semantic search performance with large corpus."""
        import time
        
        # Create a large corpus
        corpus = [f"Document {i} content about various topics" for i in range(1000)]
        
        # Mock embeddings for performance test
        with patch('backend.blocks.semantic_search.embed_text') as mock_embed:
            mock_embed.return_value = [0.1] * 1536  # Standard embedding dimension
            
            start_time = time.time()
            
            # Simulate semantic search
            query = "test query"
            query_embedding = await embed_text(query, "text-embedding-3-small")
            
            # Calculate similarities (in practice, this would be vectorized)
            results = []
            for text in corpus[:100]:  # Test with subset for performance
                text_embedding = await embed_text(text, "text-embedding-3-small")
                similarity = calculate_cosine_similarity(query_embedding, text_embedding)
                results.append({"text": text, "similarity": similarity})
            
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Performance assertion (should complete in reasonable time)
            assert duration < 5.0  # 5 seconds for 100 documents
            assert len(results) == 100


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v", "-k", "test_semantic_search_basic"])
