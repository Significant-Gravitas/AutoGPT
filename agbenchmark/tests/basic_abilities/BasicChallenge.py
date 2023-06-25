import pytest
from agbenchmark.Challenge import Challenge


@pytest.mark.run(order=1)
@pytest.mark.basic
class BasicChallenge(Challenge):
    pass
