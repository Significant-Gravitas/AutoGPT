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


@pytest.mark.asyncio
async def test_add(LocalCache, config, mock_embed_with_ada):
    cache = LocalCache(config)
    await cache.add("test")
    assert cache.data.texts == ["test"]
    assert cache.data.embeddings.shape == (1, EMBED_DIM)


@pytest.mark.asyncio
async def test_clear(LocalCache, config, mock_embed_with_ada):
    cache = LocalCache(config)
    assert cache.data.texts == []
    assert cache.data.embeddings.shape == (0, EMBED_DIM)

    await cache.add("test")
    assert cache.data.texts == ["test"]
    assert cache.data.embeddings.shape == (1, EMBED_DIM)

    cache.clear()
    assert cache.data.texts == []
    assert cache.data.embeddings.shape == (0, EMBED_DIM)


@pytest.mark.asyncio
async def test_get(LocalCache, config, mock_embed_with_ada):
    cache = LocalCache(config)
    assert await cache.get("test") == []

    await cache.add("test")
    assert await cache.get("test") == ["test"]


@pytest.mark.asyncio
@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
async def test_get_relevant(LocalCache, config) -> None:
    cache = LocalCache(config)
    text1 = "Sample text 1"
    text2 = "Sample text 2"
    await cache.add(text1)
    await cache.add(text2)

    result = await cache.get_relevant(text1, 1)
    assert result == [text1]


@pytest.mark.asyncio
async def test_get_stats(LocalCache, config, mock_embed_with_ada) -> None:
    cache = LocalCache(config)
    text = "Sample text"
    await cache.add(text)
    stats = cache.get_stats()
    assert stats == (1, cache.data.embeddings.shape)
