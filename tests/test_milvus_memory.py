# sourcery skip: snake-case-functions
"""Tests for the MilvusMemory class."""
import os
import sys

import pytest

from autogpt.memory.milvus import MilvusMemory


@pytest.fixture
def milvus_memory():
    def mock_config():
        """Mock the config object for testing purposes."""
        return type(
            "MockConfig",
            (object,),
            {
                "debug_mode": False,
                "continuous_mode": False,
                "speak_mode": False,
                "milvus_collection": "autogpt",
                "milvus_addr": "localhost:19530",
                "milvus_secure": False,
                "milvus_username": None,
                "milvus_password": None,
            },
        )

    cfg = mock_config()
    memory = MilvusMemory(cfg)
    return memory


@pytest.mark.skip("Skipping because it requires a Milvus server running to be tested")
def test_add(milvus_memory):
    text = "Sample text"
    milvus_memory.clear()
    milvus_memory.add(text)
    result = milvus_memory.get(text)
    assert result == [text]


@pytest.mark.skip("Skipping because it requires a Milvus server running to be tested")
def test_clear(milvus_memory):
    milvus_memory.clear()
    assert milvus_memory.collection.num_entities == 0


@pytest.mark.skip("Skipping because it requires a Milvus server running to be tested")
def test_get(milvus_memory):
    text = "Sample text"
    milvus_memory.clear()
    milvus_memory.add(text)
    result = milvus_memory.get(text)
    assert result == [text]


@pytest.mark.skip("Skipping because it requires a Milvus server running to be tested")
def test_get_relevant(milvus_memory):
    text1 = "Sample text 1"
    text2 = "Sample text 2"
    milvus_memory.clear()
    milvus_memory.add(text1)
    milvus_memory.add(text2)
    result = milvus_memory.get_relevant(text1, 1)
    assert result == [text1]


@pytest.mark.skip("Skipping because it requires a Milvus server running to be tested")
def test_get_stats(milvus_memory):
    text = "Sample text"
    milvus_memory.clear()
    milvus_memory.add(text)
    stats = milvus_memory.get_stats()
    assert len(stats) == 15
