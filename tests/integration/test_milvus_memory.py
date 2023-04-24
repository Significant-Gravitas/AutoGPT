# sourcery skip: snake-case-functions
"""Tests for the MilvusMemory class."""
import pytest

pytest.importorskip("pymilvus", "2.2.0", "Pymilvus is not installed")

from autogpt.memory.milvus import MilvusMemory


@pytest.fixture(scope="module")
def mock_config():
    """Mock the config object for testing purposes."""
    # Return a mock config object with the required attributes
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


@pytest.mark.integration_test
class TestMilvusMemory:
    """Tests for the MilvusMemory class."""

    @pytest.fixture(scope="class")
    def memory(self, mock_config) -> MilvusMemory:
        return MilvusMemory(mock_config)

    def test_add(self, memory: MilvusMemory) -> None:
        """Test adding a text to the cache"""
        text = "Sample text"
        memory.clear()
        memory.add(text)
        result = memory.get(text)
        assert result == [text]

    def test_clear(self, memory: MilvusMemory) -> None:
        """Test clearing the cache"""
        memory.clear()
        assert memory.collection.num_entities == 0

    def test_get(self, memory: MilvusMemory) -> None:
        """Test getting a text from the cache"""
        text = "Sample text"
        memory.clear()
        memory.add(text)
        result = memory.get(text)
        assert result == [text]

    def test_get_relevant(self, memory: MilvusMemory) -> None:
        """Test getting relevant texts from the cache"""
        text1 = "Sample text 1"
        text2 = "Sample text 2"
        memory.clear()
        memory.add(text1)
        memory.add(text2)
        result = memory.get_relevant(text1, 1)
        assert result == [text1]

    def test_get_stats(self, memory: MilvusMemory) -> None:
        """Test getting the cache stats"""
        text = "Sample text"
        memory.clear()
        memory.add(text)
        stats = memory.get_stats()
        assert len(stats) == 15
