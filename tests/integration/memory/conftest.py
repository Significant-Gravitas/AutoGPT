import numpy
import pytest
from pytest_mock import MockerFixture

import autogpt.memory.vector.utils as memory_utils
from autogpt.config.config import Config
from autogpt.llm.providers.openai import OPEN_AI_EMBEDDING_MODELS
from autogpt.memory.vector.memory_item import MemoryItem
from autogpt.memory.vector.utils import Embedding


@pytest.fixture
def embeding_dimension(config: Config):
    return OPEN_AI_EMBEDDING_MODELS[config.embedding_model].embedding_dimensions


@pytest.fixture
def mock_get_embedding(mocker: MockerFixture, embeding_dimension: int):
    mocker.patch.object(
        memory_utils,
        "get_embedding",
        return_value=[0.0255] * embeding_dimension,
    )


@pytest.fixture
def mock_embedding(embeding_dimension: int) -> Embedding:
    return numpy.full((1, embeding_dimension), 0.0255, numpy.float32)[0]


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
