# sourcery skip: snake-case-functions
"""Tests for JSONFileMemory class"""
import orjson
import pytest
from forge.config.config import Config
from forge.file_storage import FileStorage

from autogpt.memory.vector import JSONFileMemory, MemoryItem


def test_json_memory_init_without_backing_file(config: Config, storage: FileStorage):
    index_file = storage.root / f"{config.memory_index}.json"

    assert not index_file.exists()
    JSONFileMemory(config)
    assert index_file.exists()
    assert index_file.read_text() == "[]"


def test_json_memory_init_with_backing_empty_file(config: Config, storage: FileStorage):
    index_file = storage.root / f"{config.memory_index}.json"
    index_file.touch()

    assert index_file.exists()
    JSONFileMemory(config)
    assert index_file.exists()
    assert index_file.read_text() == "[]"


def test_json_memory_init_with_backing_invalid_file(
    config: Config, storage: FileStorage
):
    index_file = storage.root / f"{config.memory_index}.json"
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
        index.get("test", config) is None
    ), "Cannot test get() because initial index is not empty"

    index.add(memory_item)
    retrieved = index.get("test", config)
    assert retrieved is not None
    assert retrieved.memory_item == memory_item


def test_json_memory_load_index(config: Config, memory_item: MemoryItem):
    index = JSONFileMemory(config)
    index.add(memory_item)

    try:
        assert index.file_path.exists(), "index was not saved to file"
        assert len(index) == 1, f"index contains {len(index)} items instead of 1"
        assert index.memories[0] == memory_item, "item in index != added mock item"
    except AssertionError as e:
        raise ValueError(f"Setting up for load_index test failed: {e}")

    index.memories = []
    index.load_index()

    assert len(index) == 1
    assert index.memories[0] == memory_item


@pytest.mark.vcr
@pytest.mark.requires_openai_api_key
def test_json_memory_get_relevant(config: Config, cached_openai_client: None) -> None:
    index = JSONFileMemory(config)
    mem1 = MemoryItem.from_text_file("Sample text", "sample.txt", config)
    mem2 = MemoryItem.from_text_file(
        "Grocery list:\n- Pancake mix", "groceries.txt", config
    )
    mem3 = MemoryItem.from_text_file(
        "What is your favorite color?", "color.txt", config
    )
    lipsum = "Lorem ipsum dolor sit amet"
    mem4 = MemoryItem.from_text_file(" ".join([lipsum] * 100), "lipsum.txt", config)
    index.add(mem1)
    index.add(mem2)
    index.add(mem3)
    index.add(mem4)

    assert index.get_relevant(mem1.raw_content, 1, config)[0].memory_item == mem1
    assert index.get_relevant(mem2.raw_content, 1, config)[0].memory_item == mem2
    assert index.get_relevant(mem3.raw_content, 1, config)[0].memory_item == mem3
    assert [mr.memory_item for mr in index.get_relevant(lipsum, 2, config)] == [
        mem4,
        mem1,
    ]


def test_json_memory_get_stats(config: Config, memory_item: MemoryItem) -> None:
    index = JSONFileMemory(config)
    index.add(memory_item)
    n_memories, n_chunks = index.get_stats()
    assert n_memories == 1
    assert n_chunks == 1
