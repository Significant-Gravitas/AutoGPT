# sourcery skip: snake-case-functions
"""Tests for LocalCache class"""
import numpy
import orjson
import pytest
from pytest_mock import MockerFixture

import autogpt.memory.context.providers.json_file as json_file_memory
from autogpt.config import Config
from autogpt.memory.context import JSONFileMemory, MemoryItem
from autogpt.memory.context.utils import Embedding
from autogpt.workspace import Workspace
from tests.utils import requires_api_key

EMBED_DIM = 1536


@pytest.fixture(autouse=True)
def cleanup_sut_singleton():
    if JSONFileMemory in JSONFileMemory._instances:
        del JSONFileMemory._instances[JSONFileMemory]


@pytest.fixture
def mock_get_embedding(mocker: MockerFixture):
    mocker.patch.object(
        json_file_memory,
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
        e_summary=mock_embedding,
        chunks=["test content"],
        e_chunks=[mock_embedding],
        metadata={},
    )


def test_json_memory_init_without_backing_file(config: Config, workspace: Workspace):
    index_file = workspace.root / f"{config.memory_index}.json"

    assert not index_file.exists()
    JSONFileMemory(config)
    assert index_file.exists()
    assert index_file.read_text() == "[]"


def test_json_memory_init_with_backing_empty_file(config: Config, workspace: Workspace):
    index_file = workspace.root / f"{config.memory_index}.json"
    index_file.touch()

    assert index_file.exists()
    JSONFileMemory(config)
    assert index_file.exists()
    assert index_file.read_text() == "[]"


def test_json_memory_init_with_backing_file(config: Config, workspace: Workspace):
    index_file = workspace.root / f"{config.memory_index}.json"
    index_file.touch()

    raw_data = {"texts": ["test"]}
    data = orjson.dumps(raw_data, option=JSONFileMemory.SAVE_OPTIONS)
    with index_file.open("wb") as f:
        f.write(data)

    assert index_file.exists()
    JSONFileMemory(config)
    assert index_file.exists()
    assert index_file.read_text() == "[]"


def test_json_memory_add(config: Config, memory_item: MemoryItem):
    index = JSONFileMemory(config)
    index.add(memory_item)
    assert index.memories[0] == memory_item


def test_json_memory_clear(config: Config, memory_item: MemoryItem):
    index = JSONFileMemory(config)
    assert index.memories == []

    index.add(memory_item)
    assert index.memories[0] == memory_item, "Cannot test clear() because add() fails"

    index.clear()
    assert index.memories == []


def test_json_memory_get(config: Config, memory_item: MemoryItem, mock_get_embedding):
    index = JSONFileMemory(config)
    assert (
        index.get("test") == None
    ), "Cannot test get() because initial index is not empty"

    index.add(memory_item)
    retrieved = index.get("test")
    assert retrieved is not None
    assert retrieved == memory_item


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_json_memory_get_relevant(config: Config) -> None:
    index = JSONFileMemory(config)
    mem1 = MemoryItem.from_text_file("Sample text", "sample.txt")
    mem2 = MemoryItem.from_text_file("Grocery list:\n- Pancake mix", "groceries.txt")
    mem3 = MemoryItem.from_text_file("What is your favorite color?", "color.txt")
    lipsum = "Lorem ipsum dolor sit amet"
    mem4 = MemoryItem.from_text_file(" ".join([lipsum] * 100), "lipsum.txt")
    index.add(mem1)
    index.add(mem2)
    index.add(mem3)
    index.add(mem4)

    assert index.get_relevant(mem1.raw_content, 1)[0].memory_item == mem1
    assert index.get_relevant(mem2.raw_content, 1)[0].memory_item == mem2
    assert index.get_relevant(mem3.raw_content, 1)[0].memory_item == mem3
    assert [mr.memory_item for mr in index.get_relevant(lipsum, 2)] == [mem4, mem1]


def test_json_memory_get_stats(config: Config, memory_item: MemoryItem) -> None:
    index = JSONFileMemory(config)
    index.add(memory_item)
    n_memories, n_chunks = index.get_stats()
    assert n_memories == 1
    assert n_chunks == 1
