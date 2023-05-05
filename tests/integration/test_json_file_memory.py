# sourcery skip: snake-case-functions
"""Tests for LocalCache class"""
import orjson
import pytest
from pytest_mock import MockerFixture

from autogpt.config import Config
from autogpt.memory.providers import JSONFileMemory
from autogpt.workspace import Workspace
from tests.utils import requires_api_key

EMBED_DIM = 1536


@pytest.fixture
def mock_get_embedding(mocker: MockerFixture):
    mocker.patch.object(
        JSONFileMemory,
        "get_embedding",
        return_value=[0.1] * EMBED_DIM,
    )


def test_init_without_backing_file(config: Config, workspace: Workspace):
    index_file = workspace.root / f"{config.memory_index}.json"

    assert not index_file.exists()
    JSONFileMemory(config)
    assert index_file.exists()
    assert index_file.read_text() == "[]"


def test_init_with_backing_empty_file(config: Config, workspace: Workspace):
    index_file = workspace.root / f"{config.memory_index}.json"
    index_file.touch()

    assert index_file.exists()
    JSONFileMemory(config)
    assert index_file.exists()
    assert index_file.read_text() == "[]"


def test_init_with_backing_file(config: Config, workspace: Workspace):
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


def test_add(config: Config, mock_get_embedding):
    index = JSONFileMemory(config)
    index.add("test")
    assert index.memories[0].raw_content == "test"
    assert len(index.memories[0].e_chunks) == 1


def test_clear(config: Config, mock_get_embedding):
    index = JSONFileMemory(config)
    assert index.memories == []

    index.add("test")
    assert index.memories[0].raw_content == "test"

    index.clear()
    assert index.memories == []


def test_get(config: Config, mock_get_embedding):
    index = JSONFileMemory(config)
    assert index.get("test") == []

    index.add("test")
    assert index.get("test").raw_content == "test"


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_get_relevant(config) -> None:
    index = JSONFileMemory(config)
    text1 = "Sample text 1"
    text2 = "Sample text 2"
    index.add(text1)
    index.add(text2)

    result = index.get_relevant(text1, 1)
    assert result[0].raw_content == text1


def test_get_stats(config: Config, mock_get_embedding) -> None:
    index = JSONFileMemory(config)
    text = "Sample text"
    index.add(text)
    stats = index.get_stats()
    assert stats == (1, 1)
