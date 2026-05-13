"""Hybrid semantic + lexical search over a user's library agents.

Used by the CoPilot ``find_library_agent`` tool's ``for_creation`` mode and
by the ``create_agent`` similarity gate. Library-agent embeddings live in
the ``UnifiedContentEmbedding`` table scoped by ``userId``; we delegate the
actual ranking to ``unified_hybrid_search`` (via the ``db_accessors.search``
shim) so we inherit its graceful degradation (lexical-only fallback when
the embedding API is unavailable) and its BM25 reranking — and so the
call works whether Prisma is connected in-process or only available via
the database-manager RPC service.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from prisma.enums import ContentType

from backend.api.features.store.hybrid_search import UnifiedSearchWeights
from backend.data.db_accessors import search

logger = logging.getLogger(__name__)

# Minimum combined relevance score for a library agent to be considered
# "functionally similar" enough to recommend before creating a new one.
# Calibrated against ``_LIBRARY_SEARCH_WEIGHTS``: with semantic-biased
# weights, a true near-duplicate scores ~0.75 (semantic ≈0.85 × 0.85
# weight) and an unrelated agent lands well below 0.5.
LIBRARY_SIMILARITY_THRESHOLD = 0.55

# Library-agent search weights differ from the default unified-search mix.
# Library agents have no categories on their metadata so ``category``
# carries no signal here. Semantic still dominates because user goals are
# free-form natural language and the keyword-extracted lexical query (see
# ``_extract_lexical_keywords``) only sees ~5 stemmed tokens.
_LIBRARY_SEARCH_WEIGHTS = UnifiedSearchWeights(
    semantic=0.85,
    lexical=0.10,
    category=0.0,
    recency=0.05,
)

# English stopwords + a few command verbs (``build``, ``create``, ``make``)
# that are common in user goals but are pure noise to a tsvector match.
# Kept short on purpose: PostgreSQL's ``english`` text-search config
# already strips most stopwords during ``to_tsvector`` — this list only
# trims tokens that the user is *likely* to type and that would
# otherwise eat slots in the keyword budget (see ``_MAX_KEYWORDS``).
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "build",
        "by",
        "can",
        "create",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "make",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "should",
        "some",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "us",
        "want",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

_MAX_KEYWORDS = 5
# Strip surrounding punctuation but keep intra-word characters (so
# ``CamelCase``, ``email-summariser`` etc. survive intact).
_TOKEN_SPLIT_RE = re.compile(r"\s+")


def _extract_lexical_keywords(text: str, max_keywords: int = _MAX_KEYWORDS) -> str:
    """Return up to ``max_keywords`` distinctive tokens from ``text``.

    ``plainto_tsquery`` AND-joins every word, so a long natural-language
    goal ("Summarize a YouTube video with timestamped bullet points and a
    topic summary from a URL input") zeroes the ``@@`` match against any
    agent description that doesn't contain *all* the terms. Feeding it a
    short keyword string ("youtube video summarize timestamped bullet")
    keeps the AND semantics but lets near-duplicate agents match.

    Stopwords are dropped, tokens shorter than 3 chars are skipped (one-
    and two-letter tokens carry little signal and inflate the AND), and
    duplicates are de-duped while preserving original order.
    """
    keywords: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_SPLIT_RE.split(text):
        token = raw.lower().strip(".,!?;:'\"()[]{}<>")
        if len(token) < 3 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return " ".join(keywords)


async def hybrid_search_library_agents(
    query: str,
    user_id: str,
    limit: int = 5,
    min_score: float = LIBRARY_SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """Search the user's library agents by hybrid relevance.

    Args:
        query: The user's goal text (free-form).
        user_id: Owner of the library agents to search.
        limit: Maximum number of matches to return.
        min_score: Minimum combined relevance to keep a match.

    Returns:
        A list of result dicts ordered by relevance. Each row contains at
        least ``content_id`` (the LibraryAgent id) and ``combined_score`` /
        ``relevance``. Returns ``[]`` when the query is empty.
    """
    query = (query or "").strip()
    if not query:
        return []

    # Semantic path keeps the full sentence (richer context for the
    # embedding); lexical path uses a stopword-stripped keyword form so
    # plainto_tsquery's AND-of-terms doesn't zero out every match.
    lexical_query = _extract_lexical_keywords(query) or query

    results, _total = await search().unified_hybrid_search(
        query=query,
        content_types=[ContentType.LIBRARY_AGENT],
        page=1,
        page_size=max(1, limit),
        min_score=min_score,
        user_id=user_id,
        weights=_LIBRARY_SEARCH_WEIGHTS,
        lexical_query=lexical_query,
    )
    return results
