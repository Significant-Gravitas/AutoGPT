#!/usr/bin/env python3
"""
Documentation Indexer

Creates a hybrid search index from markdown documentation files:
- Local embeddings via sentence-transformers (all-MiniLM-L6-v2)
- BM25 index for lexical search
- PageRank scores based on internal link graph
- Title index for fast title matching

Based on ZIM-Plus indexing architecture.

Usage:
    python index.py [--docs-dir ./content] [--output index.bin] [--json]
"""

import argparse
import hashlib
import pickle
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Optional imports with graceful fallback
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Run: pip install sentence-transformers")

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
    print("Warning: rank_bm25 not installed. Run: pip install rank-bm25")

# Default embedding model (compatible with Transformers.js)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Chunk:
    """A chunk of text from a document."""
    doc_path: str           # Relative path to source document
    doc_title: str          # Document title (from first H1 or filename)
    chunk_id: int           # Chunk index within document
    text: str               # Chunk text content
    heading: str            # Current heading context
    start_char: int         # Start position in original doc
    end_char: int           # End position in original doc
    embedding: Optional[np.ndarray] = None  # OpenAI embedding vector


@dataclass
class Document:
    """A markdown document."""
    path: str               # Relative path from docs root
    title: str              # Document title
    content: str            # Raw markdown content
    headings: list[str] = field(default_factory=list)  # All headings
    outgoing_links: list[str] = field(default_factory=list)  # Links to other docs


@dataclass
class SearchIndex:
    """Complete search index structure."""
    # Metadata
    version: str = "1.0.0"
    docs_hash: str = ""     # Hash of all docs for cache invalidation
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Document data
    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)

    # Embeddings matrix (num_chunks x embedding_dim)
    embeddings: Optional[np.ndarray] = None

    # BM25 index (serialized)
    bm25_corpus: list[list[str]] = field(default_factory=list)

    # PageRank scores per document
    pagerank: dict[str, float] = field(default_factory=dict)

    # Title inverted index: word -> list of (doc_idx, score)
    title_index: dict[str, list[tuple[int, float]]] = field(default_factory=dict)

    # Path to doc index mapping
    path_to_idx: dict[str, int] = field(default_factory=dict)


# ============================================================================
# Markdown Parsing
# ============================================================================

def extract_title(content: str, filename: str) -> str:
    """Extract document title from first H1 heading or filename."""
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return filename.replace('.md', '').replace('-', ' ').replace('_', ' ').title()


def extract_headings(content: str) -> list[str]:
    """Extract all headings from markdown."""
    headings = []
    for match in re.finditer(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE):
        level = len(match.group(1))
        text = match.group(2).strip()
        headings.append(f"{'#' * level} {text}")
    return headings


def extract_links(content: str, current_path: str) -> list[str]:
    """Extract internal markdown links, normalized to relative paths."""
    links = []
    # Match [text](path) but not external URLs
    for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', content):
        link_path = match.group(2)
        # Skip external links, anchors, and images
        if link_path.startswith(('http://', 'https://', '#', 'mailto:')):
            continue
        if link_path.endswith(('.png', '.jpg', '.gif', '.svg')):
            continue

        # Normalize the path relative to docs root
        # Handle relative paths like ../foo.md or ./bar.md
        current_dir = Path(current_path).parent
        normalized = (current_dir / link_path).as_posix()
        # Remove ./ prefix and normalize
        normalized = re.sub(r'^\./', '', normalized)
        # Ensure .md extension
        if not normalized.endswith('.md'):
            normalized += '.md' if '.' not in Path(normalized).name else ''
        links.append(normalized)

    return links


def parse_document(path: Path, docs_root: Path) -> Document:
    """Parse a markdown document."""
    content = path.read_text(encoding='utf-8')
    rel_path = path.relative_to(docs_root).as_posix()

    return Document(
        path=rel_path,
        title=extract_title(content, path.name),
        content=content,
        headings=extract_headings(content),
        outgoing_links=extract_links(content, rel_path)
    )


# ============================================================================
# Chunking
# ============================================================================

def chunk_markdown(
    content: str,
    doc_path: str,
    doc_title: str,
    chunk_size: int = 6000,
    chunk_overlap: int = 200
) -> list[Chunk]:
    """
    Chunk markdown content with heading awareness.

    Strategy:
    1. Split by headings to preserve semantic boundaries
    2. Further split large sections by paragraphs
    3. Merge small sections to reach target chunk size
    4. Add overlap between chunks for context continuity
    """
    chunks = []

    # Split content into sections by headings
    sections = []
    current_heading = doc_title
    current_text = []
    current_start = 0

    lines = content.split('\n')
    char_pos = 0

    for line in lines:
        # Check if this is a heading
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)

        if heading_match:
            # Save previous section if not empty
            if current_text:
                section_text = '\n'.join(current_text)
                sections.append({
                    'heading': current_heading,
                    'text': section_text,
                    'start': current_start,
                    'end': char_pos
                })

            # Start new section
            current_heading = heading_match.group(2).strip()
            current_text = [line]
            current_start = char_pos
        else:
            current_text.append(line)

        char_pos += len(line) + 1  # +1 for newline

    # Don't forget the last section
    if current_text:
        section_text = '\n'.join(current_text)
        sections.append({
            'heading': current_heading,
            'text': section_text,
            'start': current_start,
            'end': char_pos
        })

    # Now merge small sections and split large ones
    chunk_id = 0
    buffer_text = ""
    buffer_heading = doc_title
    buffer_start = 0

    for section in sections:
        section_text = section['text'].strip()
        if not section_text:
            continue

        # If adding this section would exceed chunk size
        if len(buffer_text) + len(section_text) > chunk_size:
            # Save current buffer as chunk if not empty
            if buffer_text.strip():
                chunks.append(Chunk(
                    doc_path=doc_path,
                    doc_title=doc_title,
                    chunk_id=chunk_id,
                    text=buffer_text.strip(),
                    heading=buffer_heading,
                    start_char=buffer_start,
                    end_char=section['start']
                ))
                chunk_id += 1

            # If section itself is too large, split it
            if len(section_text) > chunk_size:
                # Split by paragraphs
                paragraphs = re.split(r'\n\n+', section_text)
                para_buffer = ""
                para_start = section['start']

                for para in paragraphs:
                    if len(para_buffer) + len(para) > chunk_size:
                        if para_buffer.strip():
                            chunks.append(Chunk(
                                doc_path=doc_path,
                                doc_title=doc_title,
                                chunk_id=chunk_id,
                                text=para_buffer.strip(),
                                heading=section['heading'],
                                start_char=para_start,
                                end_char=para_start + len(para_buffer)
                            ))
                            chunk_id += 1
                        para_buffer = para
                        para_start = para_start + len(para_buffer)
                    else:
                        para_buffer += "\n\n" + para if para_buffer else para

                # Remaining paragraph buffer becomes new buffer
                buffer_text = para_buffer
                buffer_heading = section['heading']
                buffer_start = para_start
            else:
                # Start new buffer with this section
                buffer_text = section_text
                buffer_heading = section['heading']
                buffer_start = section['start']
        else:
            # Add section to buffer
            buffer_text += "\n\n" + section_text if buffer_text else section_text
            if not buffer_heading or buffer_heading == doc_title:
                buffer_heading = section['heading']

    # Don't forget the last buffer
    if buffer_text.strip():
        chunks.append(Chunk(
            doc_path=doc_path,
            doc_title=doc_title,
            chunk_id=chunk_id,
            text=buffer_text.strip(),
            heading=buffer_heading,
            start_char=buffer_start,
            end_char=len(content)
        ))

    # Add overlap by prepending context from previous chunk
    if chunk_overlap > 0 and len(chunks) > 1:
        for i in range(1, len(chunks)):
            prev_text = chunks[i-1].text
            if len(prev_text) > chunk_overlap:
                # Find a good break point (end of sentence or paragraph)
                overlap_text = prev_text[-chunk_overlap:]
                # Try to start at a sentence boundary
                sentence_match = re.search(r'[.!?]\s+', overlap_text)
                if sentence_match:
                    overlap_text = overlap_text[sentence_match.end():]
                chunks[i].text = f"...{overlap_text}\n\n{chunks[i].text}"

    return chunks


# ============================================================================
# Embeddings
# ============================================================================

def create_embeddings_local(
    chunks: list[Chunk],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32
) -> np.ndarray:
    """
    Create embeddings using sentence-transformers (local model).

    Uses all-MiniLM-L6-v2 by default which is compatible with Transformers.js
    for client-side query embedding.
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        raise RuntimeError(
            "sentence-transformers not installed. Run: pip install sentence-transformers"
        )

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Creating embeddings for {len(chunks)} chunks...")
    texts = [chunk.text for chunk in chunks]

    # Encode with progress
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings.astype(np.float32)


def create_embeddings_openai(
    chunks: list[Chunk],
    model: str = "text-embedding-3-small",
    batch_size: int = 100
) -> np.ndarray:
    """Create OpenAI embeddings for all chunks (requires API key)."""
    if not HAS_OPENAI:
        raise RuntimeError("OpenAI library not installed")

    client = OpenAI()
    embeddings = []

    print(f"Creating OpenAI embeddings for {len(chunks)} chunks...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.text[:8000] for c in batch]

        response = client.embeddings.create(
            model=model,
            input=texts
        )

        for embedding_data in response.data:
            embeddings.append(embedding_data.embedding)

        print(f"  Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    return np.array(embeddings, dtype=np.float32)


# ============================================================================
# BM25 Index
# ============================================================================

def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    # Lowercase and extract words
    text = text.lower()
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    # Extract words
    words = re.findall(r'\b[a-z][a-z0-9_-]*\b', text)
    # Remove very short words and stopwords
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below',
                 'between', 'under', 'again', 'further', 'then', 'once',
                 'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                 'neither', 'not', 'only', 'own', 'same', 'than', 'too',
                 'very', 'just', 'also', 'now', 'here', 'there', 'when',
                 'where', 'why', 'how', 'all', 'each', 'every', 'both',
                 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                 'any', 'this', 'that', 'these', 'those', 'it', 'its'}
    return [w for w in words if len(w) > 2 and w not in stopwords]


def build_bm25_corpus(chunks: list[Chunk]) -> list[list[str]]:
    """Build tokenized corpus for BM25."""
    return [tokenize(chunk.text) for chunk in chunks]


# ============================================================================
# PageRank
# ============================================================================

def build_link_graph(documents: list[Document]) -> dict[str, list[str]]:
    """Build adjacency list from document links."""
    # Create path lookup
    valid_paths = {doc.path for doc in documents}

    graph = defaultdict(list)
    for doc in documents:
        for link in doc.outgoing_links:
            # Normalize link path
            normalized = link.lstrip('./')
            if normalized in valid_paths:
                graph[doc.path].append(normalized)

    return dict(graph)


def compute_pagerank(
    documents: list[Document],
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> dict[str, float]:
    """
    Compute PageRank scores using power iteration.

    Args:
        documents: List of documents with outgoing_links
        damping: Damping factor (probability of following a link)
        max_iterations: Maximum iterations for convergence
        tolerance: Convergence threshold

    Returns:
        Dictionary mapping document paths to PageRank scores
    """
    n = len(documents)
    if n == 0:
        return {}

    # Build path to index mapping
    path_to_idx = {doc.path: i for i, doc in enumerate(documents)}
    valid_paths = set(path_to_idx.keys())

    # Build adjacency matrix
    # out_links[i] = list of indices that document i links to
    out_links = []
    for doc in documents:
        links = []
        for link in doc.outgoing_links:
            normalized = link.lstrip('./')
            if normalized in valid_paths:
                links.append(path_to_idx[normalized])
        out_links.append(links)

    # Initialize PageRank scores uniformly
    pr = np.ones(n) / n

    # Power iteration
    for iteration in range(max_iterations):
        new_pr = np.ones(n) * (1 - damping) / n

        for i in range(n):
            if out_links[i]:
                # Distribute PageRank to outgoing links
                contribution = damping * pr[i] / len(out_links[i])
                for j in out_links[i]:
                    new_pr[j] += contribution
            else:
                # Dangling node: distribute to all nodes
                new_pr += damping * pr[i] / n

        # Check convergence
        diff = np.abs(new_pr - pr).sum()
        pr = new_pr

        if diff < tolerance:
            print(f"  PageRank converged after {iteration + 1} iterations")
            break

    # Normalize to [0, 1] range
    pr = (pr - pr.min()) / (pr.max() - pr.min() + 1e-10)

    return {documents[i].path: float(pr[i]) for i in range(n)}


# ============================================================================
# Title Index
# ============================================================================

def build_title_index(documents: list[Document]) -> dict[str, list[tuple[int, float]]]:
    """
    Build inverted index for title search.

    Returns:
        Dictionary mapping words to list of (doc_index, score) tuples
    """
    index = defaultdict(list)

    for doc_idx, doc in enumerate(documents):
        # Tokenize title
        words = tokenize(doc.title)
        word_set = set(words)

        for word in word_set:
            # Score based on word position and frequency
            score = 1.0
            if words and words[0] == word:
                score += 0.5  # Bonus for first word
            index[word].append((doc_idx, score))

    return dict(index)


# ============================================================================
# Main Indexing Pipeline
# ============================================================================

def compute_docs_hash(docs_dir: Path) -> str:
    """Compute hash of all doc files for cache invalidation."""
    hasher = hashlib.md5()
    for path in sorted(docs_dir.rglob('*.md')):
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def build_index(
    docs_dir: Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 6000,
    chunk_overlap: int = 200,
    skip_embeddings: bool = False,
    use_openai: bool = False
) -> SearchIndex:
    """
    Build complete search index from documentation directory.

    Args:
        docs_dir: Path to documentation directory
        embedding_model: Embedding model to use (default: all-MiniLM-L6-v2)
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        skip_embeddings: Skip embedding generation (for testing)
        use_openai: Use OpenAI embeddings instead of local model

    Returns:
        Complete SearchIndex ready for search
    """
    print(f"Building index from {docs_dir}")

    # Find all markdown files
    md_files = list(docs_dir.rglob('*.md'))
    print(f"Found {len(md_files)} markdown files")

    if not md_files:
        raise ValueError(f"No markdown files found in {docs_dir}")

    # Parse all documents
    print("Parsing documents...")
    documents = [parse_document(path, docs_dir) for path in md_files]

    # Create path to index mapping
    path_to_idx = {doc.path: i for i, doc in enumerate(documents)}

    # Chunk all documents
    print("Chunking documents...")
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_markdown(
            doc.content,
            doc.path,
            doc.title,
            chunk_size,
            chunk_overlap
        )
        all_chunks.extend(doc_chunks)
    print(f"Created {len(all_chunks)} chunks")

    # Build BM25 corpus
    print("Building BM25 index...")
    bm25_corpus = build_bm25_corpus(all_chunks)

    # Compute PageRank
    print("Computing PageRank...")
    pagerank = compute_pagerank(documents)

    # Build title index
    print("Building title index...")
    title_index = build_title_index(documents)

    # Create embeddings
    embeddings = None
    embedding_dim = DEFAULT_EMBEDDING_DIM
    if not skip_embeddings:
        if use_openai:
            if HAS_OPENAI:
                embeddings = create_embeddings_openai(all_chunks, embedding_model)
                embedding_dim = embeddings.shape[1]
            else:
                print("Skipping embeddings (openai not installed)")
        else:
            if HAS_SENTENCE_TRANSFORMERS:
                embeddings = create_embeddings_local(all_chunks, embedding_model)
                embedding_dim = embeddings.shape[1]
            else:
                print("Skipping embeddings (sentence-transformers not installed)")

    # Compute docs hash
    docs_hash = compute_docs_hash(docs_dir)

    # Build final index
    index = SearchIndex(
        version="1.0.0",
        docs_hash=docs_hash,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        documents=documents,
        chunks=all_chunks,
        embeddings=embeddings,
        bm25_corpus=bm25_corpus,
        pagerank=pagerank,
        title_index=title_index,
        path_to_idx=path_to_idx
    )

    return index


def save_index(index: SearchIndex, output_path: Path) -> None:
    """Save index to binary file."""
    print(f"Saving index to {output_path}")

    # Convert embeddings to float16 for space savings
    if index.embeddings is not None:
        index.embeddings = index.embeddings.astype(np.float16)

    with open(output_path, 'wb') as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Index saved ({size_mb:.2f} MB)")


def save_index_json(index: SearchIndex, output_path: Path) -> None:
    """
    Save index to JSON format for client-side JavaScript search.

    The JSON structure is optimized for browser loading:
    - Chunks with text, metadata, and embeddings
    - BM25 vocabulary and document frequencies
    - PageRank scores
    - Title index
    """
    import json
    import base64

    print(f"Saving JSON index to {output_path}")

    # Build chunks array
    chunks_data = []
    for i, chunk in enumerate(index.chunks):
        chunk_data = {
            "id": i,
            "doc": chunk.doc_path,
            "title": chunk.doc_title,
            "heading": chunk.heading,
            "text": chunk.text,
        }

        # Add embedding if available (as base64 float32)
        if index.embeddings is not None:
            emb = index.embeddings[i].astype(np.float32)
            chunk_data["emb"] = base64.b64encode(emb.tobytes()).decode('ascii')

        chunks_data.append(chunk_data)

    # Build BM25 data
    # Calculate IDF for each term
    bm25_data = {}
    if index.bm25_corpus:
        # Build vocabulary with document frequencies
        doc_freq = {}
        for doc_tokens in index.bm25_corpus:
            seen = set()
            for token in doc_tokens:
                if token not in seen:
                    doc_freq[token] = doc_freq.get(token, 0) + 1
                    seen.add(token)

        n_docs = len(index.bm25_corpus)
        bm25_data = {
            "n_docs": n_docs,
            "avgdl": sum(len(d) for d in index.bm25_corpus) / max(n_docs, 1),
            "df": doc_freq,  # document frequency per term
            "doc_lens": [len(d) for d in index.bm25_corpus],
        }

    # Build output structure
    output = {
        "version": index.version,
        "embedding_model": index.embedding_model,
        "embedding_dim": index.embedding_dim,
        "chunks": chunks_data,
        "bm25": bm25_data,
        "pagerank": index.pagerank,
        "title_index": {k: list(v) for k, v in index.title_index.items()},
    }

    # Write JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, separators=(',', ':'))  # Compact JSON

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"JSON index saved ({size_mb:.2f} MB)")


def load_index(index_path: Path) -> SearchIndex:
    """Load index from binary file."""
    with open(index_path, 'rb') as f:
        index = pickle.load(f)

    # Convert embeddings back to float32 for computation
    if index.embeddings is not None:
        index.embeddings = index.embeddings.astype(np.float32)

    return index


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Index documentation for hybrid search"
    )
    parser.add_argument(
        '--docs-dir',
        type=Path,
        default=Path('./content/platform'),
        help='Path to documentation directory (default: ./content/platform)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./index.bin'),
        help='Output index file path (default: ./index.bin)'
    )
    parser.add_argument(
        '--json-output',
        type=Path,
        default=None,
        help='Output path for JSON index (default: same as --output with .json extension)'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f'Embedding model (default: {DEFAULT_EMBEDDING_MODEL})'
    )
    parser.add_argument(
        '--use-openai',
        action='store_true',
        help='Use OpenAI embeddings instead of local sentence-transformers'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=6000,
        help='Chunk size in characters (default: 6000)'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help='Chunk overlap in characters (default: 200)'
    )
    parser.add_argument(
        '--skip-embeddings',
        action='store_true',
        help='Skip embedding generation (for testing)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Also output JSON format for client-side JavaScript search'
    )
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Only output JSON format (skip binary pickle)'
    )

    args = parser.parse_args()

    if not args.docs_dir.exists():
        print(f"Error: Documentation directory not found: {args.docs_dir}")
        sys.exit(1)

    try:
        index = build_index(
            args.docs_dir,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            skip_embeddings=args.skip_embeddings,
            use_openai=args.use_openai
        )

        # Save binary format unless json-only
        if not args.json_only:
            save_index(index, args.output)

        # Save JSON format if requested
        if args.json or args.json_only:
            json_path = args.json_output if args.json_output else args.output.with_suffix('.json')
            save_index_json(index, json_path)

        print("\nIndex Statistics:")
        print(f"  Documents: {len(index.documents)}")
        print(f"  Chunks: {len(index.chunks)}")
        print(f"  Embeddings: {'Yes' if index.embeddings is not None else 'No'}")
        print(f"  PageRank scores: {len(index.pagerank)}")
        print(f"  Title index terms: {len(index.title_index)}")

    except Exception as e:
        print(f"Error building index: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
