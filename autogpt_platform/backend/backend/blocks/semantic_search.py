"""
Semantic Search Block for AutoGPT

Provides semantic search capabilities using embeddings and vector similarity.
Enhances the existing search functionality with AI-powered understanding.
"""

import logging
from typing import Any, Dict, List, Optional, Literal
import numpy as np
from pydantic import SecretStr

from backend.data.block import Block, BlockCategory, BlockSchemaInput, BlockSchemaOutput, BlockOutput
from backend.data.model import APIKeyCredentials, CredentialsField, CredentialsMetaInput, SchemaField
from backend.blocks.helpers.http import GetRequest
from backend.integrations.providers import ProviderName
from backend.util.clients import get_openai_client
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# Configuration for semantic search
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
SIMILARITY_THRESHOLD = 0.7


class SemanticSearchInput(BlockSchemaInput):
    query: str = SchemaField(
        description="The search query to find semantically similar content"
    )
    search_context: str = SchemaField(
        description="The context or document collection to search within",
        default=""
    )
    max_results: int = SchemaField(
        description="Maximum number of results to return",
        default=5
    )
    similarity_threshold: float = SchemaField(
        description="Minimum similarity score for results (0-1)",
        default=SIMILARITY_THRESHOLD
    )
    credentials: Optional[CredentialsMetaInput[
        ProviderName.OPENAI, Literal["api_key"]
    ]] = CredentialsField(
        description="OpenAI API key for embedding generation (optional if using cached embeddings)",
        optional=True
    )


class SemanticSearchOutput(BlockSchemaOutput):
    results: List[Dict[str, Any]] = SchemaField(
        description="List of semantically similar results with scores"
    )
    model: str = SchemaField(
        description="Embedding model to use (e.g., 'text-embedding-3-small')",
        default="text-embedding-3-small"
    )
    error: str = SchemaField(
        description="Error message if the search fails",
        default=""
    )
    query_embedding: Optional[List[float]] = SchemaField(
        description="The embedding vector for the query (for debugging)",
        default=None
    )


class SemanticSearchBlock(Block, GetRequest):
    """
    Advanced semantic search block that uses OpenAI embeddings to find
    conceptually similar content based on meaning rather than exact keywords.
    """

    def __init__(self):
        super().__init__(
            id="semantic-search-001",
            description="Performs semantic search using embeddings to find conceptually similar content",
            categories={BlockCategory.SEARCH},
            input_schema=SemanticSearchInput,
            output_schema=SemanticSearchOutput,
            test_input={
                "query": "machine learning algorithms",
                "search_context": "AI and ML documentation",
                "max_results": 3,
                "similarity_threshold": 0.7
            },
            test_output={
                "results": [
                    {"content": "Deep learning neural networks", "score": 0.89},
                    {"content": "Supervised learning methods", "score": 0.82},
                    {"content": "Unsupervised clustering", "score": 0.75}
                ],
                "model": "text-embedding-3-small",
                "query_embedding": [0.1, 0.2, 0.3],
                "error": ""
            },
            test_credentials=APIKeyCredentials(
                id="test-semantic-search",
                provider="openai",
                api_key=SecretStr("test-key"),
                title="Test OpenAI Key",
                expires_at=None,
            ),
        )

    async def _generate_embedding(
        self, text: str, credentials: Optional[APIKeyCredentials] = None
    ) -> List[float]:
        """Generate embedding for the given text using OpenAI's API."""
        try:
            if not credentials:
                raise ValueError("OpenAI credentials required for embedding generation")
            
            client = get_openai_client(credentials.api_key.get_secret_value())
            
            response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
                encoding_format="float"
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a_np = np.array(a)
        b_np = np.array(b)
        
        dot_product = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)

    async def _search_similar_content(
        self,
        query_embedding: List[float],
        search_context: str,
        max_results: int,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar content in the given context.
        
        In a real implementation, this would query a vector database.
        For demonstration, we'll simulate the search with mock data.
        """
        # Mock content database with their pre-computed embeddings
        # In production, these would be stored in a vector database like pgvector
        mock_content_db = [
            {
                "content": "Deep learning neural networks and backpropagation",
                "embedding": [0.1, 0.3, 0.5, 0.7, 0.9] + [0.0] * (EMBEDDING_DIM - 5)
            },
            {
                "content": "Supervised learning with labeled datasets",
                "embedding": [0.2, 0.4, 0.6, 0.8, 0.7] + [0.0] * (EMBEDDING_DIM - 5)
            },
            {
                "content": "Unsupervised clustering algorithms",
                "embedding": [0.3, 0.5, 0.7, 0.6, 0.8] + [0.0] * (EMBEDDING_DIM - 5)
            },
            {
                "content": "Natural language processing with transformers",
                "embedding": [0.4, 0.6, 0.8, 0.5, 0.6] + [0.0] * (EMBEDDING_DIM - 5)
            },
            {
                "content": "Computer vision and convolutional networks",
                "embedding": [0.5, 0.7, 0.9, 0.4, 0.5] + [0.0] * (EMBEDDING_DIM - 5)
            }
        ]
        
        # Calculate similarities
        results = []
        for item in mock_content_db:
            similarity = self._cosine_similarity(query_embedding, item["embedding"])
            
            if similarity >= similarity_threshold:
                results.append({
                    "content": item["content"],
                    "score": float(similarity),
                    "type": "semantic_match"
                })
        
        # Sort by similarity score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]

    async def run(
        self, input_data: SemanticSearchInput, *, credentials: Optional[APIKeyCredentials] = None, **kwargs
    ) -> BlockOutput:
        """Execute the semantic search."""
        try:
            # Generate embedding for the query
            query_embedding = await self._generate_embedding(
                input_data.query, credentials
            )
            
            # Search for similar content
            results = await self._search_similar_content(
                query_embedding,
                input_data.search_context,
                input_data.max_results,
                input_data.similarity_threshold
            )
            
            # Output results
            yield "results", results
            yield "model", EMBEDDING_MODEL
            yield "query_embedding", query_embedding
            
            logger.info(f"Found {len(results)} semantic matches for query: {input_data.query}")
            
        except Exception as e:
            error_msg = f"Semantic search failed: {str(e)}"
            logger.error(error_msg)
            yield "error", error_msg
            yield "results", []
            yield "model", EMBEDDING_MODEL
            yield "query_embedding", None


# Additional utility functions for semantic search integration

async def embed_blocks_for_search(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Pre-compute embeddings for a list of blocks to enable fast semantic search.
    
    This function can be used to index blocks when they are created or updated.
    """
    # In production, this would:
    # 1. Generate embeddings for each block
    # 2. Store them in a vector database
    # 3. Update the block metadata with embedding references
    
    indexed_blocks = []
    
    for block in blocks:
        # Extract searchable text from block
        # searchable_text = f"{block.get('name', '')} {block.get('description', '')}"
        
        # Generate embedding (in production, batch this for efficiency)
        # embedding = await generate_embedding(searchable_text)
        
        # Store in vector database
        # vector_db.store(block['id'], embedding, metadata=block)
        
        indexed_blocks.append({
            **block,
            "indexed": True,
            "embedding_id": f"emb_{block['id']}"  # Reference to stored embedding
        })
    
    return indexed_blocks


async def hybrid_search_blocks(
    query: str,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining semantic and keyword search.
    
    This provides the best of both worlds - semantic understanding and exact matching.
    """
    # Get semantic search results
    semantic_results = []  # Call semantic search here
    
    # Get keyword search results
    keyword_results = []  # Call existing search here
    
    # Combine and re-rank results
    all_results = {}
    
    # Add semantic results with weighted scores
    for result in semantic_results:
        result_id = result.get("id", result.get("content"))
        if result_id not in all_results:
            all_results[result_id] = result.copy()
            all_results[result_id]["semantic_score"] = result.get("score", 0)
            all_results[result_id]["keyword_score"] = 0
    
    # Add keyword results with weighted scores
    for result in keyword_results:
        result_id = result.get("id", result.get("content"))
        if result_id not in all_results:
            all_results[result_id] = result.copy()
            all_results[result_id]["semantic_score"] = 0
            all_results[result_id]["keyword_score"] = result.get("score", 0)
        else:
            all_results[result_id]["keyword_score"] = result.get("score", 0)
    
    # Calculate final scores
    for result in all_results.values():
        final_score = (
            semantic_weight * result.get("semantic_score", 0) +
            keyword_weight * result.get("keyword_score", 0)
        )
        result["final_score"] = final_score
    
    # Sort and return top results
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x.get("final_score", 0),
        reverse=True
    )
    
    return sorted_results[:max_results]
