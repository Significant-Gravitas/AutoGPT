"""Tool for searching platform documentation."""

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from backend.server.v2.chat.model import ChatSession
from backend.server.v2.chat.tools.base import BaseTool
from backend.server.v2.chat.tools.models import (
    DocSearchResult,
    DocSearchResultsResponse,
    ErrorResponse,
    NoResultsResponse,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)

# Documentation base URL
DOCS_BASE_URL = "https://docs.agpt.co/platform"

# Path to the JSON index file (relative to this file)
INDEX_PATH = Path(__file__).parent / "docs_index.json"


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    text = text.lower()
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)
    # Extract words
    words = re.findall(r"\b[a-z][a-z0-9_-]*\b", text)
    # Remove very short words and stopwords
    stopwords = {
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
        "both",
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
    }
    return [w for w in words if len(w) > 2 and w not in stopwords]


class DocSearchIndex:
    """Lightweight documentation search index using BM25."""

    def __init__(self, index_path: Path):
        self.chunks: list[dict] = []
        self.bm25_data: dict = {}
        self._loaded = False
        self._index_path = index_path

    def load(self) -> bool:
        """Load the index from JSON file."""
        if self._loaded:
            return True

        if not self._index_path.exists():
            logger.warning(f"Documentation index not found at {self._index_path}")
            return False

        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.chunks = data.get("chunks", [])
            self.bm25_data = data.get("bm25", {})
            self._loaded = True
            logger.info(f"Loaded documentation index with {len(self.chunks)} chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to load documentation index: {e}")
            return False

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the index using BM25."""
        if not self._loaded and not self.load():
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # BM25 parameters
        k1 = 1.5
        b = 0.75
        n_docs = self.bm25_data.get("n_docs", len(self.chunks))
        avgdl = self.bm25_data.get("avgdl", 100)
        df = self.bm25_data.get("df", {})
        doc_lens = self.bm25_data.get("doc_lens", [100] * len(self.chunks))

        scores = []
        for i, chunk in enumerate(self.chunks):
            # Tokenize chunk text
            chunk_tokens = tokenize(chunk.get("text", ""))
            doc_len = doc_lens[i] if i < len(doc_lens) else len(chunk_tokens)

            # Calculate BM25 score
            score = 0.0
            for token in query_tokens:
                if token not in df:
                    continue

                # Term frequency in this document
                tf = chunk_tokens.count(token)
                if tf == 0:
                    continue

                # IDF
                doc_freq = df.get(token, 0)
                idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

                # BM25 score component
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
                score += idf * numerator / denominator

            # Boost for title/heading matches
            title = chunk.get("title", "").lower()
            heading = chunk.get("heading", "").lower()
            for token in query_tokens:
                if token in title:
                    score *= 1.5
                if token in heading:
                    score *= 1.2

            scores.append((i, score))

        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        seen_sections = set()
        for idx, score in scores:
            if score <= 0:
                continue

            chunk = self.chunks[idx]
            section_key = (chunk.get("doc", ""), chunk.get("heading", ""))

            # Deduplicate by section
            if section_key in seen_sections:
                continue
            seen_sections.add(section_key)

            results.append(
                {
                    "title": chunk.get("title", ""),
                    "path": chunk.get("doc", ""),
                    "heading": chunk.get("heading", ""),
                    "text": chunk.get("text", ""),  # Full text for LLM comprehension
                    "score": score,
                }
            )

            if len(results) >= top_k:
                break

        return results


# Global index instance (lazy loaded)
_search_index: DocSearchIndex | None = None


def get_search_index() -> DocSearchIndex:
    """Get or create the search index singleton."""
    global _search_index
    if _search_index is None:
        _search_index = DocSearchIndex(INDEX_PATH)
    return _search_index


class SearchDocsTool(BaseTool):
    """Tool for searching AutoGPT platform documentation."""

    @property
    def name(self) -> str:
        return "search_platform_docs"

    @property
    def description(self) -> str:
        return (
            "Search the AutoGPT platform documentation for information about "
            "how to use the platform, create agents, configure blocks, "
            "set up integrations, and more. Use this when users ask questions "
            "about how to do something with AutoGPT."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query describing what the user wants to learn about. "
                        "Use keywords like 'blocks', 'agents', 'credentials', 'API', etc."
                    ),
                },
            },
            "required": ["query"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """Search documentation for the query.

        Args:
            user_id: User ID (may be anonymous)
            session: Chat session
            query: Search query

        Returns:
            DocSearchResultsResponse: List of matching documentation sections
            NoResultsResponse: No results found
            ErrorResponse: Error message
        """
        query = kwargs.get("query", "").strip()
        session_id = session.session_id

        if not query:
            return ErrorResponse(
                message="Please provide a search query",
                session_id=session_id,
            )

        try:
            index = get_search_index()
            results = index.search(query, top_k=5)

            if not results:
                return NoResultsResponse(
                    message=f"No documentation found for '{query}'. Try different keywords.",
                    session_id=session_id,
                    suggestions=[
                        "Try more general terms like 'blocks', 'agents', 'setup'",
                        "Check the documentation at docs.agpt.co",
                    ],
                )

            # Convert to response format
            doc_results = []
            for r in results:
                # Build documentation URL
                path = r["path"]
                if path.endswith(".md"):
                    path = path[:-3]  # Remove .md extension
                doc_url = f"{DOCS_BASE_URL}/{path}"

                full_text = r["text"]
                doc_results.append(
                    DocSearchResult(
                        title=r["title"],
                        path=r["path"],
                        section=r["heading"],
                        snippet=(
                            full_text[:300] + "..."
                            if len(full_text) > 300
                            else full_text
                        ),
                        content=full_text,  # Full text for LLM to read and understand
                        score=round(r["score"], 3),
                        doc_url=doc_url,
                    )
                )

            return DocSearchResultsResponse(
                message=(
                    f"Found {len(doc_results)} relevant documentation sections. "
                    "Use these to help answer the user's question. "
                    "Include links to the documentation when helpful."
                ),
                results=doc_results,
                count=len(doc_results),
                query=query,
                session_id=session_id,
            )

        except Exception as e:
            logger.error(f"Error searching documentation: {e}", exc_info=True)
            return ErrorResponse(
                message="Failed to search documentation. Please try again.",
                error=str(e),
                session_id=session_id,
            )
