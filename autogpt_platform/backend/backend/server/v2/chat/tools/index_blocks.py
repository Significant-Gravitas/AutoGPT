#!/usr/bin/env python3
"""
Block Indexer for Hybrid Search

Creates a hybrid search index from blocks:
- OpenAI embeddings (text-embedding-3-small)
- BM25 index for lexical search
- Name index for title matching boost

Supports incremental updates by tracking content hashes.

Usage:
    python -m backend.server.v2.chat.tools.index_blocks [--force]
"""

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai not installed. Run: pip install openai")

# Default embedding model (OpenAI)
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIM = 1536

# Output path (relative to this file)
INDEX_PATH = Path(__file__).parent / "blocks_index.json"

# Stopwords for tokenization
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
    "block",  # Too common in block context
}


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    text = text.lower()
    # Remove code blocks if any
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)
    # Extract words (including camelCase split)
    # First, split camelCase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Extract words
    words = re.findall(r"\b[a-z][a-z0-9_-]*\b", text)
    # Remove very short words and stopwords
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def build_searchable_text(block: Any) -> str:
    """Build searchable text from block attributes."""
    parts = []

    # Block name (split camelCase for better tokenization)
    name = block.name
    # Split camelCase: GetCurrentTimeBlock -> Get Current Time Block
    name_split = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    parts.append(name_split)

    # Description
    if block.description:
        parts.append(block.description)

    # Categories
    for category in block.categories:
        parts.append(category.name)

    # Input schema field names and descriptions
    try:
        input_schema = block.input_schema.jsonschema()
        if "properties" in input_schema:
            for field_name, field_info in input_schema["properties"].items():
                parts.append(field_name)
                if "description" in field_info:
                    parts.append(field_info["description"])
    except Exception:
        pass

    # Output schema field names
    try:
        output_schema = block.output_schema.jsonschema()
        if "properties" in output_schema:
            for field_name in output_schema["properties"]:
                parts.append(field_name)
    except Exception:
        pass

    return " ".join(parts)


def compute_content_hash(text: str) -> str:
    """Compute MD5 hash of text for change detection."""
    return hashlib.md5(text.encode()).hexdigest()


def load_existing_index(index_path: Path) -> dict[str, Any] | None:
    """Load existing index if present."""
    if not index_path.exists():
        return None

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load existing index: {e}")
        return None


def create_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 100,
) -> np.ndarray:
    """Create embeddings using OpenAI API."""
    if not HAS_OPENAI:
        raise RuntimeError("openai not installed. Run: pip install openai")

    # Import here to satisfy type checker
    from openai import OpenAI

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)
    embeddings = []

    print(f"Creating embeddings for {len(texts)} texts using {model_name}...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Truncate texts to max token limit (8191 tokens for text-embedding-3-small)
        # Roughly 4 chars per token, so ~32000 chars max
        batch = [text[:32000] for text in batch]

        response = client.embeddings.create(
            model=model_name,
            input=batch,
        )

        for embedding_data in response.data:
            embeddings.append(embedding_data.embedding)

        print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

    return np.array(embeddings, dtype=np.float32)


def build_bm25_data(
    blocks_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build BM25 metadata from block data."""
    # Tokenize all searchable texts
    tokenized_docs = []
    for block in blocks_data:
        tokens = tokenize(block["searchable_text"])
        tokenized_docs.append(tokens)

    # Calculate document frequencies
    doc_freq: dict[str, int] = {}
    for tokens in tokenized_docs:
        seen = set()
        for token in tokens:
            if token not in seen:
                doc_freq[token] = doc_freq.get(token, 0) + 1
                seen.add(token)

    n_docs = len(tokenized_docs)
    doc_lens = [len(d) for d in tokenized_docs]
    avgdl = sum(doc_lens) / max(n_docs, 1)

    return {
        "n_docs": n_docs,
        "avgdl": avgdl,
        "df": doc_freq,
        "doc_lens": doc_lens,
    }


def build_name_index(
    blocks_data: list[dict[str, Any]],
) -> dict[str, list[list[int | float]]]:
    """Build inverted index for name search boost."""
    index: dict[str, list[list[int | float]]] = defaultdict(list)

    for idx, block in enumerate(blocks_data):
        # Tokenize block name
        name_tokens = tokenize(block["name"])
        seen = set()

        for i, token in enumerate(name_tokens):
            if token in seen:
                continue
            seen.add(token)

            # Score: first token gets higher weight
            score = 1.5 if i == 0 else 1.0
            index[token].append([idx, score])

    return dict(index)


def build_block_index(
    force_rebuild: bool = False,
    output_path: Path = INDEX_PATH,
) -> dict[str, Any]:
    """
    Build the block search index.

    Args:
        force_rebuild: If True, rebuild all embeddings even if unchanged
        output_path: Path to save the index

    Returns:
        The generated index dictionary
    """
    # Import here to avoid circular imports
    from backend.blocks import load_all_blocks

    print("Loading all blocks...")
    all_blocks = load_all_blocks()
    print(f"Found {len(all_blocks)} blocks")

    # Load existing index for incremental updates
    existing_index = None if force_rebuild else load_existing_index(output_path)
    existing_blocks: dict[str, dict[str, Any]] = {}

    if existing_index:
        print(
            f"Loaded existing index with {len(existing_index.get('blocks', []))} blocks"
        )
        for block in existing_index.get("blocks", []):
            existing_blocks[block["id"]] = block

    # Process each block
    blocks_data: list[dict[str, Any]] = []
    blocks_needing_embedding: list[tuple[int, str]] = []  # (index, searchable_text)

    for block_id, block_cls in all_blocks.items():
        try:
            block = block_cls()

            # Skip disabled blocks
            if block.disabled:
                continue

            searchable_text = build_searchable_text(block)
            content_hash = compute_content_hash(searchable_text)

            block_data = {
                "id": block.id,
                "name": block.name,
                "description": block.description,
                "categories": [cat.name for cat in block.categories],
                "searchable_text": searchable_text,
                "content_hash": content_hash,
                "emb": None,  # Will be filled later
            }

            # Check if we can reuse existing embedding
            if (
                block.id in existing_blocks
                and existing_blocks[block.id].get("content_hash") == content_hash
                and existing_blocks[block.id].get("emb")
            ):
                # Reuse existing embedding
                block_data["emb"] = existing_blocks[block.id]["emb"]
            else:
                # Need new embedding
                blocks_needing_embedding.append((len(blocks_data), searchable_text))

            blocks_data.append(block_data)

        except Exception as e:
            logger.warning(f"Failed to process block {block_id}: {e}")
            continue

    print(f"Processed {len(blocks_data)} blocks")
    print(f"Blocks needing new embeddings: {len(blocks_needing_embedding)}")

    # Create embeddings for new/changed blocks
    if blocks_needing_embedding and HAS_OPENAI:
        texts_to_embed = [text for _, text in blocks_needing_embedding]
        try:
            embeddings = create_embeddings(texts_to_embed)

            # Assign embeddings to blocks
            for i, (block_idx, _) in enumerate(blocks_needing_embedding):
                emb = embeddings[i].astype(np.float32)
                # Encode as base64
                blocks_data[block_idx]["emb"] = base64.b64encode(emb.tobytes()).decode(
                    "ascii"
                )
        except Exception as e:
            print(f"Warning: Failed to create embeddings: {e}")
    elif blocks_needing_embedding:
        print(
            "Warning: Cannot create embeddings (openai not installed or OPENAI_API_KEY not set)"
        )

    # Build BM25 data
    print("Building BM25 index...")
    bm25_data = build_bm25_data(blocks_data)

    # Build name index
    print("Building name index...")
    name_index = build_name_index(blocks_data)

    # Build final index
    index = {
        "version": "1.0.0",
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blocks": blocks_data,
        "bm25": bm25_data,
        "name_index": name_index,
    }

    # Save index
    print(f"Saving index to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"))

    size_kb = output_path.stat().st_size / 1024
    print(f"Index saved ({size_kb:.1f} KB)")

    # Print statistics
    print("\nIndex Statistics:")
    print(f"  Blocks indexed: {len(blocks_data)}")
    print(f"  BM25 vocabulary size: {len(bm25_data['df'])}")
    print(f"  Name index terms: {len(name_index)}")
    print(f"  Embeddings: {'Yes' if any(b.get('emb') for b in blocks_data) else 'No'}")

    return index


def main():
    parser = argparse.ArgumentParser(description="Build hybrid search index for blocks")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild all embeddings even if unchanged",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=INDEX_PATH,
        help=f"Output index file path (default: {INDEX_PATH})",
    )

    args = parser.parse_args()

    try:
        build_block_index(
            force_rebuild=args.force,
            output_path=args.output,
        )
    except Exception as e:
        print(f"Error building index: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
