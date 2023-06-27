import pytest
from agbenchmark.Challenge import Challenge
from agbenchmark.challenges.define_task_types import ChallengeData
from abc import abstractmethod


@pytest.mark.basic
class BasicChallenge(Challenge):
    pass
