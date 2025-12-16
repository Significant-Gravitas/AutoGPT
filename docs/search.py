#!/usr/bin/env python3
"""
Documentation Search

Hybrid search combining:
- Semantic search (OpenAI embeddings + cosine similarity)
- Lexical search (BM25)
- Authority ranking (PageRank)
- Title matching
- Content quality signals

Based on ZIM-Plus search architecture with tunable weights.

Usage:
    python search.py "your query" [--index index.bin] [--top-k 10]
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from index import SearchIndex, Chunk, Document, load_index, tokenize


# ============================================================================
# Search Configuration
# ============================================================================

@dataclass
class SearchWeights:
    """
    Hybrid search weight configuration.

    Based on ZIM-Plus reranking signals:
    - semantic: Cosine similarity from embeddings
    - title_match: Query terms appearing in title
    - url_path_match: Query terms appearing in URL/path
    - bm25: Sparse lexical matching score
    - content_quality: Penalizes TOC/nav/boilerplate chunks
    - pagerank: Link authority score
    - position_boost: Prefers earlier chunks in document

    All weights should sum to 1.0 for interpretability.
    """
    semantic: float = 0.30
    title_match: float = 0.20
    url_path_match: float = 0.15
    bm25: float = 0.15
    content_quality: float = 0.10
    pagerank: float = 0.05
    position_boost: float = 0.05

    # Diversity penalty: max chunks per document
    max_chunks_per_doc: int = 2

    def validate(self) -> None:
        """Ensure weights are valid."""
        total = (self.semantic + self.title_match + self.url_path_match +
                 self.bm25 + self.content_quality + self.pagerank +
                 self.position_boost)
        if abs(total - 1.0) > 0.01:
            print(f"Warning: Weights sum to {total:.3f}, not 1.0")


# Default weights (tuned for documentation search)
DEFAULT_WEIGHTS = SearchWeights()

# Alternative weight presets for different use cases
WEIGHT_PRESETS = {
    "semantic_heavy": SearchWeights(
        semantic=0.50, title_match=0.15, url_path_match=0.10,
        bm25=0.10, content_quality=0.05, pagerank=0.05, position_boost=0.05
    ),
    "keyword_heavy": SearchWeights(
        semantic=0.20, title_match=0.20, url_path_match=0.15,
        bm25=0.30, content_quality=0.05, pagerank=0.05, position_boost=0.05
    ),
    "authority_heavy": SearchWeights(
        semantic=0.25, title_match=0.15, url_path_match=0.10,
        bm25=0.15, content_quality=0.10, pagerank=0.20, position_boost=0.05
    ),
}


# ============================================================================
# Search Result
# ============================================================================

@dataclass
class SearchResult:
    """A single search result with scoring breakdown."""
    chunk: Chunk
    score: float  # Final combined score

    # Individual signal scores (for debugging/tuning)
    semantic_score: float = 0.0
    title_score: float = 0.0
    path_score: float = 0.0
    bm25_score: float = 0.0
    quality_score: float = 0.0
    pagerank_score: float = 0.0
    position_score: float = 0.0

    def __str__(self) -> str:
        return (
            f"[{self.score:.3f}] {self.chunk.doc_title}\n"
            f"  Path: {self.chunk.doc_path}\n"
            f"  Heading: {self.chunk.heading}\n"
            f"  Scores: sem={self.semantic_score:.2f} title={self.title_score:.2f} "
            f"path={self.path_score:.2f} bm25={self.bm25_score:.2f} "
            f"qual={self.quality_score:.2f} pr={self.pagerank_score:.2f} "
            f"pos={self.position_score:.2f}"
        )


# ============================================================================
# Search Engine
# ============================================================================

class HybridSearcher:
    """
    Hybrid search engine combining multiple ranking signals.
    """

    def __init__(
        self,
        index: SearchIndex,
        weights: SearchWeights = DEFAULT_WEIGHTS,
        openai_client: Optional[OpenAI] = None
    ):
        self.index = index
        self.weights = weights
        self.weights.validate()

        # Detect if index was built with local embeddings (sentence-transformers)
        self.use_local_embeddings = index.embedding_model in [
            "all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"
        ]

        # Initialize local embedding model if needed
        self.local_model = None
        if self.use_local_embeddings and HAS_SENTENCE_TRANSFORMERS:
            self.local_model = SentenceTransformer(index.embedding_model)

        # Initialize OpenAI client for query embedding (only if not using local)
        self.openai_client = None
        if not self.use_local_embeddings:
            self.openai_client = openai_client
            if openai_client is None and HAS_OPENAI:
                self.openai_client = OpenAI()

        # Build BM25 index from stored corpus
        self.bm25 = None
        if HAS_BM25 and index.bm25_corpus:
            self.bm25 = BM25Okapi(index.bm25_corpus)

        # Precompute normalized embeddings for faster cosine similarity
        self.normalized_embeddings = None
        if index.embeddings is not None:
            norms = np.linalg.norm(index.embeddings, axis=1, keepdims=True)
            self.normalized_embeddings = index.embeddings / (norms + 1e-10)

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """Get embedding for search query."""
        # Use local model if available
        if self.local_model is not None:
            embedding = self.local_model.encode(query, convert_to_numpy=True)
            return embedding.astype(np.float32)

        # Fall back to OpenAI
        if self.openai_client is None:
            return None

        response = self.openai_client.embeddings.create(
            model=self.index.embedding_model,
            input=query
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def compute_semantic_scores(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all chunks."""
        if self.normalized_embeddings is None:
            return np.zeros(len(self.index.chunks))

        # Normalize query embedding
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        # Cosine similarity via dot product of normalized vectors
        similarities = self.normalized_embeddings @ query_norm

        # Normalize to [0, 1] range
        similarities = (similarities + 1) / 2  # cosine ranges from -1 to 1

        return similarities

    def compute_bm25_scores(self, query_tokens: list[str]) -> np.ndarray:
        """Compute BM25 scores for all chunks."""
        if self.bm25 is None:
            return np.zeros(len(self.index.chunks))

        scores = self.bm25.get_scores(query_tokens)

        # Normalize to [0, 1] range
        if scores.max() > 0:
            scores = scores / scores.max()

        return scores

    def compute_title_scores(self, query_tokens: list[str]) -> np.ndarray:
        """Compute title match scores for all chunks."""
        scores = np.zeros(len(self.index.chunks))
        query_set = set(query_tokens)

        for chunk_idx, chunk in enumerate(self.index.chunks):
            title_tokens = set(tokenize(chunk.doc_title))

            # Exact matches
            exact_matches = len(query_set & title_tokens)

            # Partial matches (substring)
            partial_matches = 0
            for qt in query_tokens:
                for tt in title_tokens:
                    if qt in tt or tt in qt:
                        partial_matches += 0.5

            # Compute score
            if query_tokens:
                scores[chunk_idx] = (exact_matches * 2 + partial_matches) / (len(query_tokens) * 2)

        return np.clip(scores, 0, 1)

    def compute_path_scores(self, query_tokens: list[str]) -> np.ndarray:
        """Compute URL/path match scores for all chunks."""
        scores = np.zeros(len(self.index.chunks))
        query_set = set(query_tokens)

        for chunk_idx, chunk in enumerate(self.index.chunks):
            # Extract path components
            path_parts = re.split(r'[/_-]', chunk.doc_path.lower())
            path_parts = [p.replace('.md', '') for p in path_parts if p]
            path_set = set(path_parts)

            # Count matches
            matches = len(query_set & path_set)

            # Partial matches
            partial = 0
            for qt in query_tokens:
                for pp in path_parts:
                    if qt in pp or pp in qt:
                        partial += 0.5

            if query_tokens:
                scores[chunk_idx] = (matches * 2 + partial) / (len(query_tokens) * 2)

        return np.clip(scores, 0, 1)

    def compute_quality_scores(self) -> np.ndarray:
        """
        Compute content quality scores.

        Penalizes:
        - TOC/navigation chunks (lots of links, little content)
        - Very short chunks
        - Chunks that are mostly code
        """
        scores = np.ones(len(self.index.chunks))

        for chunk_idx, chunk in enumerate(self.index.chunks):
            text = chunk.text
            penalty = 0.0

            # Penalize TOC-like content (many links)
            link_count = len(re.findall(r'\[([^\]]+)\]\([^)]+\)', text))
            if link_count > 10:
                penalty += 0.3

            # Penalize very short chunks
            if len(text) < 200:
                penalty += 0.2

            # Penalize chunks that are mostly code
            code_blocks = re.findall(r'```[\s\S]*?```', text)
            code_length = sum(len(b) for b in code_blocks)
            if len(text) > 0 and code_length / len(text) > 0.8:
                penalty += 0.2

            # Penalize index/navigation pages
            if chunk.doc_path.endswith('index.md'):
                penalty += 0.1

            scores[chunk_idx] = max(0, 1 - penalty)

        return scores

    def compute_pagerank_scores(self) -> np.ndarray:
        """Get PageRank scores for all chunks (by document)."""
        scores = np.zeros(len(self.index.chunks))

        for chunk_idx, chunk in enumerate(self.index.chunks):
            scores[chunk_idx] = self.index.pagerank.get(chunk.doc_path, 0.0)

        return scores

    def compute_position_scores(self) -> np.ndarray:
        """Compute position boost (prefer earlier chunks in document)."""
        scores = np.zeros(len(self.index.chunks))

        # Group chunks by document
        doc_chunks = {}
        for chunk_idx, chunk in enumerate(self.index.chunks):
            if chunk.doc_path not in doc_chunks:
                doc_chunks[chunk.doc_path] = []
            doc_chunks[chunk.doc_path].append(chunk_idx)

        for doc_path, chunk_indices in doc_chunks.items():
            n = len(chunk_indices)
            for i, chunk_idx in enumerate(chunk_indices):
                # Earlier chunks get higher scores (linear decay)
                scores[chunk_idx] = 1 - (i / max(n, 1))

        return scores

    def search(
        self,
        query: str,
        top_k: int = 10,
        apply_diversity: bool = True
    ) -> list[SearchResult]:
        """
        Perform hybrid search.

        Args:
            query: Search query string
            top_k: Number of results to return
            apply_diversity: Apply diversity penalty (max chunks per doc)

        Returns:
            List of SearchResult objects sorted by score
        """
        # Tokenize query
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Compute all signal scores
        semantic_scores = np.zeros(len(self.index.chunks))
        if self.normalized_embeddings is not None and (self.local_model or self.openai_client):
            query_embedding = self.embed_query(query)
            if query_embedding is not None:
                semantic_scores = self.compute_semantic_scores(query_embedding)

        bm25_scores = self.compute_bm25_scores(query_tokens)
        title_scores = self.compute_title_scores(query_tokens)
        path_scores = self.compute_path_scores(query_tokens)
        quality_scores = self.compute_quality_scores()
        pagerank_scores = self.compute_pagerank_scores()
        position_scores = self.compute_position_scores()

        # Combine scores using weights
        w = self.weights
        combined_scores = (
            w.semantic * semantic_scores +
            w.title_match * title_scores +
            w.url_path_match * path_scores +
            w.bm25 * bm25_scores +
            w.content_quality * quality_scores +
            w.pagerank * pagerank_scores +
            w.position_boost * position_scores
        )

        # Create results
        results = []
        for chunk_idx in range(len(self.index.chunks)):
            results.append(SearchResult(
                chunk=self.index.chunks[chunk_idx],
                score=combined_scores[chunk_idx],
                semantic_score=semantic_scores[chunk_idx],
                title_score=title_scores[chunk_idx],
                path_score=path_scores[chunk_idx],
                bm25_score=bm25_scores[chunk_idx],
                quality_score=quality_scores[chunk_idx],
                pagerank_score=pagerank_scores[chunk_idx],
                position_score=position_scores[chunk_idx]
            ))

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)

        # Apply diversity penalty
        if apply_diversity:
            results = self._apply_diversity(results, top_k)

        return results[:top_k]

    def _apply_diversity(
        self,
        results: list[SearchResult],
        target_k: int
    ) -> list[SearchResult]:
        """
        Deduplicate results from the same document unless they point to
        different sections (headings) within the page.

        Logic:
        1. Only keep one result per unique (doc_path, heading) combination
        2. Additionally limit total chunks per document to max_chunks_per_doc
        """
        seen_sections = set()  # (doc_path, heading) tuples
        doc_counts = {}        # doc_path -> count
        filtered = []

        for result in results:
            doc_path = result.chunk.doc_path
            heading = result.chunk.heading
            section_key = (doc_path, heading)

            # Skip if we've already seen this exact section
            if section_key in seen_sections:
                continue

            # Also enforce max chunks per document
            doc_count = doc_counts.get(doc_path, 0)
            if doc_count >= self.weights.max_chunks_per_doc:
                continue

            # Keep this result
            seen_sections.add(section_key)
            doc_counts[doc_path] = doc_count + 1
            filtered.append(result)

            if len(filtered) >= target_k * 2:  # Get extra for buffer
                break

        return filtered

    def search_title_only(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Fallback search using only title index and PageRank.
        Useful when embeddings aren't available.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Score documents by title match
        doc_scores = {}
        for token in query_tokens:
            if token in self.index.title_index:
                for doc_idx, score in self.index.title_index[token]:
                    doc_path = self.index.documents[doc_idx].path
                    doc_scores[doc_path] = doc_scores.get(doc_path, 0) + score

        # Boost by PageRank
        for doc_path in doc_scores:
            pr = self.index.pagerank.get(doc_path, 0.0)
            doc_scores[doc_path] *= (1 + pr)

        # Get top documents and their first chunks
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_path, score in sorted_docs[:top_k]:
            # Find first chunk of this document
            for chunk in self.index.chunks:
                if chunk.doc_path == doc_path:
                    results.append(SearchResult(
                        chunk=chunk,
                        score=score,
                        title_score=score
                    ))
                    break

        return results


# ============================================================================
# Reciprocal Rank Fusion (Alternative scoring method)
# ============================================================================

def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60
) -> list[tuple[int, float]]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion.

    RRF is an alternative to weighted linear combination that's
    less sensitive to score scale differences.

    Args:
        rankings: List of rankings (each is list of chunk indices)
        k: RRF parameter (default 60 works well)

    Returns:
        List of (chunk_idx, rrf_score) tuples sorted by score
    """
    scores = {}

    for ranking in rankings:
        for rank, chunk_idx in enumerate(ranking):
            if chunk_idx not in scores:
                scores[chunk_idx] = 0
            scores[chunk_idx] += 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class RRFSearcher(HybridSearcher):
    """
    Alternative searcher using Reciprocal Rank Fusion instead of
    weighted linear combination.
    """

    def search(
        self,
        query: str,
        top_k: int = 10,
        apply_diversity: bool = True
    ) -> list[SearchResult]:
        """Search using RRF fusion."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Get individual rankings
        rankings = []

        # Semantic ranking
        if self.normalized_embeddings is not None and (self.local_model or self.openai_client):
            query_embedding = self.embed_query(query)
            if query_embedding is not None:
                scores = self.compute_semantic_scores(query_embedding)
                rankings.append(np.argsort(scores)[::-1].tolist())

        # BM25 ranking
        if self.bm25:
            scores = self.compute_bm25_scores(query_tokens)
            rankings.append(np.argsort(scores)[::-1].tolist())

        # Title ranking
        scores = self.compute_title_scores(query_tokens)
        rankings.append(np.argsort(scores)[::-1].tolist())

        if not rankings:
            return []

        # Fuse rankings
        fused = reciprocal_rank_fusion(rankings)

        # Build results
        results = []
        for chunk_idx, score in fused[:top_k * 3]:  # Extra for diversity
            chunk = self.index.chunks[chunk_idx]
            results.append(SearchResult(
                chunk=chunk,
                score=score
            ))

        # Apply diversity
        if apply_diversity:
            results = self._apply_diversity(results, top_k)

        return results[:top_k]


# ============================================================================
# CLI
# ============================================================================

def format_result(result: SearchResult, show_text: bool = True) -> str:
    """Format a search result for display."""
    lines = [
        f"\n{'='*60}",
        f"Score: {result.score:.3f}",
        f"Title: {result.chunk.doc_title}",
        f"Path:  {result.chunk.doc_path}",
        f"Section: {result.chunk.heading}",
    ]

    if show_text:
        # Truncate text for display
        text = result.chunk.text[:500]
        if len(result.chunk.text) > 500:
            text += "..."
        lines.append(f"\n{text}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Search documentation using hybrid search"
    )
    parser.add_argument(
        'query',
        type=str,
        help='Search query'
    )
    parser.add_argument(
        '--index',
        type=Path,
        default=Path('./index.bin'),
        help='Path to index file (default: ./index.bin)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results (default: 5)'
    )
    parser.add_argument(
        '--weights',
        type=str,
        choices=['default', 'semantic_heavy', 'keyword_heavy', 'authority_heavy'],
        default='default',
        help='Weight preset (default: default)'
    )
    parser.add_argument(
        '--rrf',
        action='store_true',
        help='Use Reciprocal Rank Fusion instead of weighted combination'
    )
    parser.add_argument(
        '--no-diversity',
        action='store_true',
        help='Disable diversity penalty'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show detailed scoring breakdown'
    )
    parser.add_argument(
        '--no-text',
        action='store_true',
        help='Hide result text snippets'
    )

    args = parser.parse_args()

    if not args.index.exists():
        print(f"Error: Index file not found: {args.index}")
        print("Run 'python index.py' first to create the index.")
        sys.exit(1)

    # Load index
    print(f"Loading index from {args.index}...")
    index = load_index(args.index)
    print(f"Loaded {len(index.chunks)} chunks from {len(index.documents)} documents")

    # Select weights
    weights = DEFAULT_WEIGHTS
    if args.weights != 'default':
        weights = WEIGHT_PRESETS[args.weights]

    # Create searcher
    SearcherClass = RRFSearcher if args.rrf else HybridSearcher
    searcher = SearcherClass(index, weights)

    # Search
    print(f"\nSearching for: '{args.query}'")
    results = searcher.search(
        args.query,
        top_k=args.top_k,
        apply_diversity=not args.no_diversity
    )

    if not results:
        print("No results found.")
        return

    print(f"\nFound {len(results)} results:")

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(format_result(result, show_text=not args.no_text))

        if args.debug:
            print(f"\nScore breakdown:")
            print(f"  Semantic:     {result.semantic_score:.3f}")
            print(f"  Title match:  {result.title_score:.3f}")
            print(f"  Path match:   {result.path_score:.3f}")
            print(f"  BM25:         {result.bm25_score:.3f}")
            print(f"  Quality:      {result.quality_score:.3f}")
            print(f"  PageRank:     {result.pagerank_score:.3f}")
            print(f"  Position:     {result.position_score:.3f}")


if __name__ == '__main__':
    main()
