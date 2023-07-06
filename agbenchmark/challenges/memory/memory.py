import pytest

from agbenchmark.challenge import Challenge


@pytest.mark.memory
class MemoryChallenge(Challenge):
    """Challenge for memory"""
