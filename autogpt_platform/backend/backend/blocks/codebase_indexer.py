"""
Codebase Indexer Block — ingests a local or remote repository and stores
embeddings in the vector memory for full-context RAG before coding tasks.

Supports: local directory paths, GitHub repo URLs (cloned automatically).
Chunks files by function/class boundaries or fixed token windows.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

logger = logging.getLogger(__name__)

# File extensions to index
INDEXABLE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
    ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".sh",
}

# Files/dirs to skip
SKIP_PATTERNS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", "target", "vendor", ".cache",
}

MAX_FILE_SIZE_BYTES = 100_000  # Skip files larger than 100KB
CHUNK_SIZE_CHARS = 2000        # Characters per chunk
CHUNK_OVERLAP_CHARS = 200      # Overlap between chunks


def _should_index_file(path: Path) -> bool:
    """Return True if the file should be indexed."""
    if path.suffix not in INDEXABLE_EXTENSIONS:
        return False
    for part in path.parts:
        if part in SKIP_PATTERNS:
            return False
    if path.stat().st_size > MAX_FILE_SIZE_BYTES:
        return False
    return True


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def _clone_repo(repo_url: str, target_dir: str) -> bool:
    """Clone a GitHub repository into target_dir. Returns True on success."""
    try:
        result = subprocess.run(
            ["git", "clone", "--depth=1", repo_url, target_dir],
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to clone {repo_url}: {e}")
        return False


class CodebaseIndexerInput(BlockSchemaInput):
    source_path: str = SchemaField(
        description=(
            "Local directory path OR GitHub URL to index. "
            "Examples: '/home/user/myproject' or 'https://github.com/user/repo'."
        )
    )
    collection_name: str = SchemaField(
        default="codebase_index",
        description="ChromaDB collection name for this codebase index.",
    )
    persist_directory: str = SchemaField(
        default="./data/chroma_db",
        description="Directory for ChromaDB persistent storage.",
    )
    chunk_size: int = SchemaField(
        default=CHUNK_SIZE_CHARS,
        description="Characters per chunk when splitting large files.",
    )
    clear_existing: bool = SchemaField(
        default=False,
        description="If True, clears the existing collection before indexing.",
    )
    file_extensions: list = SchemaField(
        default_factory=list,
        description=(
            "Optional list of file extensions to index (e.g., ['.py', '.ts']). "
            "If empty, uses the default set of common code extensions."
        ),
    )


class CodebaseIndexerOutput(BlockSchemaOutput):
    files_indexed: int = SchemaField(description="Number of files indexed.")
    chunks_stored: int = SchemaField(description="Total chunks stored in vector DB.")
    collection_name: str = SchemaField(description="ChromaDB collection used.")
    status: str = SchemaField(description="Status message.")


class CodebaseIndexerBlock(Block):
    """
    Indexes a codebase into vector memory for full-context RAG.

    Upload a local repo path or GitHub URL. The agent will chunk all source files,
    generate embeddings, and store them in ChromaDB. Subsequent coding tasks can
    then retrieve relevant context with semantic search.
    """

    class Input(CodebaseIndexerInput):
        pass

    class Output(CodebaseIndexerOutput):
        pass

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-a7b8-9012-cdef-345678901234",
            description=(
                "Index a local or GitHub repository into vector memory for "
                "full-context RAG. Chunks source files and stores embeddings in ChromaDB."
            ),
            categories={BlockCategory.AI, BlockCategory.DATA},
            input_schema=CodebaseIndexerBlock.Input,
            output_schema=CodebaseIndexerBlock.Output,
            test_input={
                "source_path": "/tmp/test_repo",
                "collection_name": "test_codebase",
                "persist_directory": "/tmp/test_chroma",
                "clear_existing": True,
            },
            test_output=[
                ("status", "Indexing complete."),
                ("files_indexed", 0),
                ("chunks_stored", 0),
                ("collection_name", "test_codebase"),
            ],
        )

    def run(self, input_data: Input, *, execution_stats=None, **kwargs) -> BlockOutput:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            yield "status", "ChromaDB not installed. Run: pip install chromadb"
            yield "files_indexed", 0
            yield "chunks_stored", 0
            yield "collection_name", input_data.collection_name
            return

        # Determine source directory
        source = input_data.source_path.strip()
        temp_dir = None

        if source.startswith("http://") or source.startswith("https://"):
            temp_dir = tempfile.mkdtemp(prefix="autogpt_codebase_")
            logger.info(f"Cloning {source} into {temp_dir}...")
            success = _clone_repo(source, temp_dir)
            if not success:
                yield "status", f"Failed to clone repository: {source}"
                yield "files_indexed", 0
                yield "chunks_stored", 0
                yield "collection_name", input_data.collection_name
                return
            source_dir = Path(temp_dir)
        else:
            source_dir = Path(source)
            if not source_dir.exists():
                yield "status", f"Source path does not exist: {source}"
                yield "files_indexed", 0
                yield "chunks_stored", 0
                yield "collection_name", input_data.collection_name
                return

        # Connect to ChromaDB
        client = chromadb.PersistentClient(
            path=input_data.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        if input_data.clear_existing:
            try:
                client.delete_collection(input_data.collection_name)
            except Exception:
                pass

        collection = client.get_or_create_collection(
            name=input_data.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Determine extensions to use
        extensions = set(input_data.file_extensions) if input_data.file_extensions else INDEXABLE_EXTENSIONS

        files_indexed = 0
        chunks_stored = 0
        batch_docs = []
        batch_ids = []
        batch_metas = []
        BATCH_SIZE = 100

        for file_path in source_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix not in extensions:
                continue
            if not _should_index_file(file_path):
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            rel_path = str(file_path.relative_to(source_dir))
            chunks = _chunk_text(content, chunk_size=input_data.chunk_size)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{rel_path}::{i}"
                import hashlib
                doc_id = hashlib.sha256(chunk_id.encode()).hexdigest()[:20]
                batch_docs.append(chunk)
                batch_ids.append(doc_id)
                batch_metas.append({
                    "file_path": rel_path,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source": str(source_dir),
                })
                chunks_stored += 1

                if len(batch_docs) >= BATCH_SIZE:
                    collection.upsert(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
                    batch_docs, batch_ids, batch_metas = [], [], []

            files_indexed += 1

        # Flush remaining
        if batch_docs:
            collection.upsert(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)

        # Cleanup temp dir
        if temp_dir:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

        yield "status", f"Indexing complete. {files_indexed} files, {chunks_stored} chunks stored."
        yield "files_indexed", files_indexed
        yield "chunks_stored", chunks_stored
        yield "collection_name", input_data.collection_name
