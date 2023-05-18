import numpy
import pytest
from pytest_mock import MockerFixture

import autogpt.memory.context.utils as memory_utils
from autogpt.memory.context.memory_item import MemoryItem
from autogpt.memory.context.utils import Embedding

EMBED_DIM = 1536


@pytest.fixture
def mock_get_embedding(mocker: MockerFixture):
    mocker.patch.object(
        memory_utils,
        "get_embedding",
        return_value=[0.0255] * EMBED_DIM,
    )


@pytest.fixture
def mock_embedding() -> Embedding:
    return numpy.full((1, EMBED_DIM), 0.0255, numpy.float32)[0]


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
