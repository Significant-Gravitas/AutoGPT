from unittest.mock import patch

import pytest
from pytest_mock import MockerFixture

from autogpt.llm.api_manager import ApiManager
from autogpt.llm.providers.openai import OPEN_AI_CHAT_MODELS, OPEN_AI_EMBEDDING_MODELS

api_manager = ApiManager()


@pytest.fixture(autouse=True)
def reset_api_manager():
    api_manager.reset()
    yield


@pytest.fixture(autouse=True)
def mock_costs(mocker: MockerFixture):
    mocker.patch.multiple(
        OPEN_AI_CHAT_MODELS["gpt-3.5-turbo"],
        prompt_token_cost=0.0013,
        completion_token_cost=0.0025,
    )
    mocker.patch.multiple(
        OPEN_AI_EMBEDDING_MODELS["text-embedding-ada-002"],
        prompt_token_cost=0.0004,
    )
    yield


class TestApiManager:
    def test_getter_methods(self):
        """Test the getter methods for total tokens, cost, and budget."""
        api_manager.update_cost(600, 1200, "gpt-3.5-turbo")
        api_manager.set_total_budget(10.0)
        assert api_manager.get_total_prompt_tokens() == 600
        assert api_manager.get_total_completion_tokens() == 1200
        assert api_manager.get_total_cost() == (600 * 0.0013 + 1200 * 0.0025) / 1000
        assert api_manager.get_total_budget() == 10.0

    @staticmethod
    def test_set_total_budget():
        """Test if setting the total budget works correctly."""
        total_budget = 10.0
        api_manager.set_total_budget(total_budget)

        assert api_manager.get_total_budget() == total_budget

    @staticmethod
    def test_update_cost_completion_model():
        """Test if updating the cost works correctly."""
        prompt_tokens = 50
        completion_tokens = 100
        model = "gpt-3.5-turbo"

        api_manager.update_cost(prompt_tokens, completion_tokens, model)

        assert api_manager.get_total_prompt_tokens() == prompt_tokens
        assert api_manager.get_total_completion_tokens() == completion_tokens
        assert (
            api_manager.get_total_cost()
            == (prompt_tokens * 0.0013 + completion_tokens * 0.0025) / 1000
        )

    @staticmethod
    def test_update_cost_embedding_model():
        """Test if updating the cost works correctly."""
        prompt_tokens = 1337
        model = "text-embedding-ada-002"

        api_manager.update_cost(prompt_tokens, 0, model)

        assert api_manager.get_total_prompt_tokens() == prompt_tokens
        assert api_manager.get_total_completion_tokens() == 0
        assert api_manager.get_total_cost() == (prompt_tokens * 0.0004) / 1000

    @staticmethod
    def test_get_models():
        """Test if getting models works correctly."""
        with patch("openai.Model.list") as mock_list_models:
            mock_list_models.return_value = {"data": [{"id": "gpt-3.5-turbo"}]}
            result = api_manager.get_models()

            assert result[0]["id"] == "gpt-3.5-turbo"
            assert api_manager.models[0]["id"] == "gpt-3.5-turbo"
