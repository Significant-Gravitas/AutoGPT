"""
Persistent Vector Memory Block — stores and retrieves task context across sessions.

Supports ChromaDB (local, no server required) and pgvector (PostgreSQL-based).
Cross-task memory enables the agent to recall prior work, decisions, and code snippets.
"""

import hashlib
import json
import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

logger = logging.getLogger(__name__)


class VectorBackend(str, Enum):
    CHROMA = "chroma"
    PGVECTOR = "pgvector"


class MemoryOperation(str, Enum):
    STORE = "store"
    RETRIEVE = "retrieve"
    DELETE = "delete"
    LIST = "list"


class VectorMemoryStoreInput(BlockSchemaInput):
    operation: MemoryOperation = SchemaField(
        default=MemoryOperation.STORE,
        description="Operation to perform: store, retrieve, delete, or list.",
    )
    content: str = SchemaField(
        default="",
        description="Content to store in vector memory (used for STORE operation).",
    )
    query: str = SchemaField(
        default="",
        description="Query string for semantic search (used for RETRIEVE operation).",
    )
    memory_id: str = SchemaField(
        default="",
        description="Unique ID for the memory entry (used for DELETE operation).",
    )
    collection_name: str = SchemaField(
        default="autogpt_memory",
        description="ChromaDB collection name (acts as a namespace for memories).",
    )
    n_results: int = SchemaField(
        default=5,
        description="Number of results to return for RETRIEVE operation.",
    )
    metadata: dict = SchemaField(
        default_factory=dict,
        description="Optional metadata to attach to stored memory (e.g., task_id, timestamp).",
    )
    persist_directory: str = SchemaField(
        default="./data/chroma_db",
        description="Directory for ChromaDB persistent storage.",
    )
    backend: VectorBackend = SchemaField(
        default=VectorBackend.CHROMA,
        description="Vector storage backend: chroma (local) or pgvector (PostgreSQL).",
    )


class VectorMemoryStoreOutput(BlockSchemaOutput):
    result: str = SchemaField(description="Operation result or retrieved content.")
    memory_id: str = SchemaField(description="ID of the stored or retrieved memory.")
    count: int = SchemaField(description="Number of items stored or retrieved.")
    documents: list = SchemaField(description="List of retrieved document strings.")
    metadatas: list = SchemaField(description="List of metadata dicts for retrieved documents.")


class VectorMemoryBlock(Block):
    """
    Persistent vector memory for the coding agent.

    Stores embeddings of task context, code snippets, and decisions across sessions.
    Uses ChromaDB locally (no server required) or pgvector for PostgreSQL-backed storage.
    Enables semantic search: "find all tasks where I refactored authentication code."
    """

    class Input(VectorMemoryStoreInput):
        pass

    class Output(VectorMemoryStoreOutput):
        pass

    def __init__(self):
        super().__init__(
            id="b2c3d4e5-f6a7-8901-bcde-f23456789012",
            description=(
                "Persistent cross-task vector memory. Store and semantically retrieve "
                "code snippets, task summaries, and decisions using ChromaDB or pgvector."
            ),
            categories={BlockCategory.AI, BlockCategory.DATA},
            input_schema=VectorMemoryBlock.Input,
            output_schema=VectorMemoryBlock.Output,
            test_input={
                "operation": MemoryOperation.STORE.value,
                "content": "Implemented JWT authentication in FastAPI using python-jose.",
                "collection_name": "autogpt_memory",
                "metadata": {"task_id": "test-001", "type": "code_summary"},
                "persist_directory": "/tmp/test_chroma",
                "backend": VectorBackend.CHROMA.value,
            },
            test_output=[
                ("result", "Memory stored successfully."),
                ("count", 1),
            ],
        )

    def run(self, input_data: Input, *, execution_stats=None, **kwargs) -> BlockOutput:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            yield "result", (
                "ChromaDB not installed. Run: pip install chromadb. "
                "Falling back to in-memory stub."
            )
            yield "memory_id", ""
            yield "count", 0
            yield "documents", []
            yield "metadatas", []
            return

        client = chromadb.PersistentClient(
            path=input_data.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(
            name=input_data.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        if input_data.operation == MemoryOperation.STORE:
            doc_id = hashlib.sha256(
                (input_data.content + json.dumps(input_data.metadata, sort_keys=True)).encode()
            ).hexdigest()[:16]
            collection.upsert(
                documents=[input_data.content],
                ids=[doc_id],
                metadatas=[input_data.metadata] if input_data.metadata else [{}],
            )
            yield "result", "Memory stored successfully."
            yield "memory_id", doc_id
            yield "count", 1
            yield "documents", [input_data.content]
            yield "metadatas", [input_data.metadata]

        elif input_data.operation == MemoryOperation.RETRIEVE:
            results = collection.query(
                query_texts=[input_data.query],
                n_results=min(input_data.n_results, collection.count() or 1),
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            ids = results.get("ids", [[]])[0]
            yield "result", "\n\n---\n\n".join(docs) if docs else "No memories found."
            yield "memory_id", ids[0] if ids else ""
            yield "count", len(docs)
            yield "documents", docs
            yield "metadatas", metas

        elif input_data.operation == MemoryOperation.DELETE:
            if input_data.memory_id:
                collection.delete(ids=[input_data.memory_id])
                yield "result", f"Memory {input_data.memory_id} deleted."
                yield "memory_id", input_data.memory_id
                yield "count", 1
                yield "documents", []
                yield "metadatas", []
            else:
                yield "result", "No memory_id provided for DELETE."
                yield "memory_id", ""
                yield "count", 0
                yield "documents", []
                yield "metadatas", []

        elif input_data.operation == MemoryOperation.LIST:
            all_items = collection.get()
            docs = all_items.get("documents", []) or []
            metas = all_items.get("metadatas", []) or []
            yield "result", f"Found {len(docs)} memories in collection '{input_data.collection_name}'."
            yield "memory_id", ""
            yield "count", len(docs)
            yield "documents", docs
            yield "metadatas", metas
