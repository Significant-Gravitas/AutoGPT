import pytest

from agbenchmark.challenge import Challenge


@pytest.mark.retrieval
class RetrievalChallenge(Challenge):
    """Challenge for information-retrieval"""
