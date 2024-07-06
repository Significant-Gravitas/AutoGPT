import pytest

from autogpt.memory.vector.memory_item import MemoryItem
from autogpt.memory.vector.utils import Embedding


@pytest.fixture
def memory_item(mock_embedding: Embedding):
    return MemoryItem(
        raw_content="test content",
        summary="test content summary",
        chunks=["test content"],
        chunk_summaries=["test content summary"],
        e_summary=mock_embedding,
        e_chunks=[mock_embedding],
        metadata={},
    )
