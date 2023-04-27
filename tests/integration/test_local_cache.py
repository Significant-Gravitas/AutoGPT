# sourcery skip: snake-case-functions
"""Tests for LocalCache class"""
import unittest

import orjson
import pytest

from autogpt.memory.local import EMBED_DIM, SAVE_OPTIONS
from autogpt.memory.local import LocalCache as LocalCache_
from tests.utils import requires_api_key


@pytest.fixture
def LocalCache():
    # Hack, real gross. Singletons are not good times.
    if LocalCache_ in LocalCache_._instances:
        del LocalCache_._instances[LocalCache_]
    return LocalCache_


@pytest.fixture
def mock_embed_with_ada(mocker):
    mocker.patch(
        "autogpt.memory.local.get_ada_embedding",
        return_value=[0.1] * EMBED_DIM,
    )


def test_init_without_backing_file(LocalCache, config, workspace):
    cache_file = workspace.root / f"{config.memory_index}.json"

    assert not cache_file.exists()
    LocalCache(config)
    assert cache_file.exists()
    assert cache_file.read_text() == "{}"


def test_init_with_backing_empty_file(LocalCache, config, workspace):
    cache_file = workspace.root / f"{config.memory_index}.json"
    cache_file.touch()

    assert cache_file.exists()
    LocalCache(config)
    assert cache_file.exists()
    assert cache_file.read_text() == "{}"


def test_init_with_backing_file(LocalCache, config, workspace):
    cache_file = workspace.root / f"{config.memory_index}.json"
    cache_file.touch()

    raw_data = {"texts": ["test"]}
    data = orjson.dumps(raw_data, option=SAVE_OPTIONS)
    with cache_file.open("wb") as f:
        f.write(data)

    assert cache_file.exists()
    LocalCache(config)
    assert cache_file.exists()
    assert cache_file.read_text() == "{}"


def test_add(LocalCache, config, mock_embed_with_ada):
    cache = LocalCache(config)
    cache.add("test")
    assert cache.data.texts == ["test"]
    assert cache.data.embeddings.shape == (1, EMBED_DIM)


def test_clear(LocalCache, config, mock_embed_with_ada):
    cache = LocalCache(config)
    assert cache.data.texts == []
    assert cache.data.embeddings.shape == (0, EMBED_DIM)

    cache.add("test")
    assert cache.data.texts == ["test"]
    assert cache.data.embeddings.shape == (1, EMBED_DIM)

    cache.clear()
    assert cache.data.texts == []
    assert cache.data.embeddings.shape == (0, EMBED_DIM)


def test_get(LocalCache, config, mock_embed_with_ada):
    cache = LocalCache(config)
    assert cache.get("test") == []

    cache.add("test")
    assert cache.get("test") == ["test"]


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_get_relevant(LocalCache, config) -> None:
    cache = LocalCache(config)
    text1 = "Sample text 1"
    text2 = "Sample text 2"
    cache.add(text1)
    cache.add(text2)

    result = cache.get_relevant(text1, 1)
    assert result == [text1]


def test_get_stats(LocalCache, config, mock_embed_with_ada) -> None:
    cache = LocalCache(config)
    text = "Sample text"
    cache.add(text)
    stats = cache.get_stats()
    assert stats == (1, cache.data.embeddings.shape)
