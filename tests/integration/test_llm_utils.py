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


@pytest.fixture()
def api_manager(mocker: MockerFixture):
    api_manager = ApiManager()
    mocker.patch.multiple(
        api_manager,
        total_prompt_tokens=0,
        total_completion_tokens=0,
        total_cost=0,
    )
    yield api_manager


@pytest.fixture()
def spy_create_embedding(mocker: MockerFixture):
    return mocker.spy(llm_utils, "create_embedding")


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_get_ada_embedding(
    config: Config, api_manager: ApiManager, spy_create_embedding: MagicMock
):
    token_cost = COSTS[config.embedding_model]["prompt"]
    llm_utils.get_ada_embedding("test")

    spy_create_embedding.assert_called_once_with("test", model=config.embedding_model)

    assert (prompt_tokens := api_manager.get_total_prompt_tokens()) == 1
    assert api_manager.get_total_completion_tokens() == 0
    assert api_manager.get_total_cost() == (prompt_tokens * token_cost) / 1000


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_get_ada_embedding_large_context(random_large_string):
    # This test should be able to mock the openai call after we have a fix.  We don't need
    # to hit the API to test the logic of the function (so not using vcr). This is a quick
    # regression test to document the issue.
    llm_utils.get_ada_embedding(random_large_string)
