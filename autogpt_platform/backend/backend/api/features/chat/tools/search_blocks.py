"""
Block Hybrid Search

Combines multiple ranking signals for block search:
- Semantic search (OpenAI embeddings + cosine similarity)
- Lexical search (BM25)
- Name matching (boost for block name matches)
- Category matching (boost for category matches)

Based on the docs search implementation.
"""

import base64
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# Path to the JSON index file
INDEX_PATH = Path(__file__).parent / "blocks_index.json"

# Stopwords for tokenization (same as index_blocks.py)
STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "dare",
    "ought",
    "used",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "then",
    "once",
    "and",
    "but",
    "or",
    "nor",
    "so",
    "yet",
    "both",
    "either",
    "neither",
    "not",
    "only",
    "own",
    "same",
    "than",
    "too",
    "very",
    "just",
    "also",
    "now",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "any",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "block",
}


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for search."""
    text = text.lower()
    # Remove code blocks if any
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)
    # Split camelCase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Extract words
    words = re.findall(r"\b[a-z][a-z0-9_-]*\b", text)
    # Remove very short words and stopwords
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


@dataclass
class SearchWeights:
    """Configuration for hybrid search signal weights."""

    semantic: float = 0.40  # Embedding similarity
    bm25: float = 0.25  # Lexical matching
    name_match: float = 0.25  # Block name matches
    category_match: float = 0.10  # Category matches


@dataclass
class BlockSearchResult:
    """A single block search result."""

    block_id: str
    name: str
    description: str
    categories: list[str]
    score: float

    # Individual signal scores (for debugging)
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    name_score: float = 0.0
    category_score: float = 0.0


class BlockSearchIndex:
    """Hybrid search index for blocks combining BM25 + embeddings."""

    def __init__(self, index_path: Path = INDEX_PATH):
        self.blocks: list[dict[str, Any]] = []
        self.bm25_data: dict[str, Any] = {}
        self.name_index: dict[str, list[list[int | float]]] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.normalized_embeddings: Optional[np.ndarray] = None
        self._loaded = False
        self._index_path = index_path
        self._embedding_model: Any = None

    def load(self) -> bool:
        """Load the index from JSON file."""
        if self._loaded:
            return True

        if not self._index_path.exists():
            logger.warning(f"Block index not found at {self._index_path}")
            return False

        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.blocks = data.get("blocks", [])
            self.bm25_data = data.get("bm25", {})
            self.name_index = data.get("name_index", {})

            # Decode embeddings from base64
            embeddings_list = []
            for block in self.blocks:
                if block.get("emb"):
                    emb_bytes = base64.b64decode(block["emb"])
                    emb = np.frombuffer(emb_bytes, dtype=np.float32)
                    embeddings_list.append(emb)
                else:
                    # No embedding, use zeros
                    dim = data.get("embedding_dim", 384)
                    embeddings_list.append(np.zeros(dim, dtype=np.float32))

            if embeddings_list:
                self.embeddings = np.stack(embeddings_list)
                # Precompute normalized embeddings for cosine similarity
                norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                self.normalized_embeddings = self.embeddings / (norms + 1e-10)

            self._loaded = True
            logger.info(f"Loaded block index with {len(self.blocks)} blocks")
            return True

        except Exception as e:
            logger.error(f"Failed to load block index: {e}")
            return False

    def _get_openai_client(self) -> Any:
        """Get OpenAI client for query embedding."""
        if self._embedding_model is None:
            try:
                from openai import OpenAI

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not set")
                    return None
                self._embedding_model = OpenAI(api_key=api_key)
            except ImportError:
                logger.warning("openai not installed")
                return None
        return self._embedding_model

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        """Embed the search query using OpenAI."""
        client = self._get_openai_client()
        if client is None:
            return None

        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=query,
            )
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to embed query: {e}")
            return None

    def _compute_semantic_scores(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all blocks."""
        if self.normalized_embeddings is None:
            return np.zeros(len(self.blocks))

        # Normalize query embedding
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        # Cosine similarity via dot product
        similarities = self.normalized_embeddings @ query_norm

        # Scale to [0, 1] (cosine ranges from -1 to 1)
        return (similarities + 1) / 2

    def _compute_bm25_scores(self, query_tokens: list[str]) -> np.ndarray:
        """Compute BM25 scores for all blocks."""
        scores = np.zeros(len(self.blocks))

        if not self.bm25_data or not query_tokens:
            return scores

        # BM25 parameters
        k1 = 1.5
        b = 0.75
        n_docs = self.bm25_data.get("n_docs", len(self.blocks))
        avgdl = self.bm25_data.get("avgdl", 100)
        df = self.bm25_data.get("df", {})
        doc_lens = self.bm25_data.get("doc_lens", [100] * len(self.blocks))

        for i, block in enumerate(self.blocks):
            # Tokenize block's searchable text
            block_tokens = tokenize(block.get("searchable_text", ""))
            doc_len = doc_lens[i] if i < len(doc_lens) else len(block_tokens)

            # Calculate BM25 score
            score = 0.0
            for token in query_tokens:
                if token not in df:
                    continue

                # Term frequency in this document
                tf = block_tokens.count(token)
                if tf == 0:
                    continue

                # IDF
                doc_freq = df.get(token, 0)
                idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

                # BM25 score component
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
                score += idf * numerator / denominator

            scores[i] = score

        # Normalize to [0, 1]
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score

        return scores

    def _compute_name_scores(self, query_tokens: list[str]) -> np.ndarray:
        """Compute name match scores using the name index."""
        scores = np.zeros(len(self.blocks))

        if not self.name_index or not query_tokens:
            return scores

        for token in query_tokens:
            if token in self.name_index:
                for block_idx, weight in self.name_index[token]:
                    if block_idx < len(scores):
                        scores[int(block_idx)] += weight

        # Also check for partial matches in block names
        for i, block in enumerate(self.blocks):
            name_lower = block.get("name", "").lower()
            for token in query_tokens:
                if token in name_lower:
                    scores[i] += 0.5

        # Normalize to [0, 1]
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score

        return scores

    def _compute_category_scores(self, query_tokens: list[str]) -> np.ndarray:
        """Compute category match scores."""
        scores = np.zeros(len(self.blocks))

        if not query_tokens:
            return scores

        for i, block in enumerate(self.blocks):
            categories = block.get("categories", [])
            category_text = " ".join(categories).lower()

            for token in query_tokens:
                if token in category_text:
                    scores[i] += 1.0

        # Normalize to [0, 1]
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score

        return scores

    def search(
        self,
        query: str,
        top_k: int = 10,
        weights: Optional[SearchWeights] = None,
    ) -> list[BlockSearchResult]:
        """
        Perform hybrid search combining multiple signals.

        Args:
            query: Search query string
            top_k: Number of results to return
            weights: Optional custom weights for signals

        Returns:
            List of BlockSearchResult sorted by score
        """
        if not self._loaded and not self.load():
            return []

        if weights is None:
            weights = SearchWeights()

        # Tokenize query
        query_tokens = tokenize(query)
        if not query_tokens:
            # Fallback: try raw query words
            query_tokens = query.lower().split()

        # Compute semantic scores
        semantic_scores = np.zeros(len(self.blocks))
        if self.normalized_embeddings is not None:
            query_embedding = self._embed_query(query)
            if query_embedding is not None:
                semantic_scores = self._compute_semantic_scores(query_embedding)

        # Compute other scores
        bm25_scores = self._compute_bm25_scores(query_tokens)
        name_scores = self._compute_name_scores(query_tokens)
        category_scores = self._compute_category_scores(query_tokens)

        # Combine scores using weights
        combined_scores = (
            weights.semantic * semantic_scores
            + weights.bm25 * bm25_scores
            + weights.name_match * name_scores
            + weights.category_match * category_scores
        )

        # Get top-k indices
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            if combined_scores[idx] <= 0:
                continue

            block = self.blocks[idx]
            results.append(
                BlockSearchResult(
                    block_id=block["id"],
                    name=block["name"],
                    description=block["description"],
                    categories=block.get("categories", []),
                    score=float(combined_scores[idx]),
                    semantic_score=float(semantic_scores[idx]),
                    bm25_score=float(bm25_scores[idx]),
                    name_score=float(name_scores[idx]),
                    category_score=float(category_scores[idx]),
                )
            )

        return results


# Global index instance (lazy loaded)
_block_search_index: Optional[BlockSearchIndex] = None


def get_block_search_index() -> BlockSearchIndex:
    """Get or create the block search index singleton."""
    global _block_search_index
    if _block_search_index is None:
        _block_search_index = BlockSearchIndex(INDEX_PATH)
    return _block_search_index
