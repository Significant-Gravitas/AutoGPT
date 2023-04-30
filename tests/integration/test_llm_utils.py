import string
from unittest.mock import MagicMock

import pytest
from numpy.random import RandomState
from pytest_mock import MockerFixture

from autogpt.config import Config
from autogpt.llm import llm_utils
from autogpt.llm.api_manager import ApiManager
from autogpt.llm.modelsinfo import COSTS
from tests.utils import requires_api_key


@pytest.fixture(scope="session")
def random_large_string():
    """Big string used to overwhelm token limits."""
    seed = 42
    n_characters = 30_000
    random = RandomState(seed)
    return "".join(random.choice(list(string.ascii_lowercase), size=n_characters))


@requires_api_key("OPENAI_API_KEY")
def test_get_ada_embedding_large_context(random_large_string):
    # This test should be able to mock the openai call after we have a fix.  We don't need
    # to hit the API to test the logic of the function (so not using vcr). This is a quick
    # regression test to document the issue.
    llm_utils.get_ada_embedding(random_large_string)
