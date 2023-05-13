from unittest.mock import MagicMock, patch

import pytest

from autogpt.llm import COSTS, ApiManager

api_manager = ApiManager()


@pytest.fixture(autouse=True)
def reset_api_manager():
    api_manager.reset()
    yield


@pytest.fixture(autouse=True)
def mock_costs():
    with patch.dict(
        COSTS,
        {
            "gpt-3.5-turbo": {"prompt": 0.002, "completion": 0.002},
            "text-embedding-ada-002": {"prompt": 0.0004, "completion": 0},
        },
        clear=True,
    ):
        yield


class TestApiManager:
    @staticmethod
    def test_create_chat_completion_debug_mode(caplog):
        """Test if debug mode logs response."""
        api_manager_debug = ApiManager(debug=True)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
        ]
        model = "gpt-3.5-turbo"

        with patch("openai.ChatCompletion.create") as mock_create:
            mock_response = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_create.return_value = mock_response

            api_manager_debug.create_chat_completion(messages, model=model)

            assert "Response" in caplog.text

    @staticmethod
    def test_create_chat_completion_empty_messages():
        """Test if empty messages result in zero tokens and cost."""
        messages = []
        model = "gpt-3.5-turbo"

        with patch("openai.ChatCompletion.create") as mock_create:
            mock_response = MagicMock()
            mock_response.usage.prompt_tokens = 0
            mock_response.usage.completion_tokens = 0
            mock_create.return_value = mock_response

            api_manager.create_chat_completion(messages, model=model)

            assert api_manager.get_total_prompt_tokens() == 0
            assert api_manager.get_total_completion_tokens() == 0
            assert api_manager.get_total_cost() == 0

    @staticmethod
    def test_create_chat_completion_valid_inputs():
        """Test if valid inputs result in correct tokens and cost."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
        ]
        model = "gpt-3.5-turbo"

        with patch("openai.ChatCompletion.create") as mock_create:
            mock_response = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_create.return_value = mock_response

            api_manager.create_chat_completion(messages, model=model)

            assert api_manager.get_total_prompt_tokens() == 10
            assert api_manager.get_total_completion_tokens() == 20
            assert api_manager.get_total_cost() == (10 * 0.002 + 20 * 0.002) / 1000

    def test_getter_methods(self):
        """Test the getter methods for total tokens, cost, and budget."""
        api_manager.update_cost(60, 120, "gpt-3.5-turbo")
        api_manager.set_total_budget(10.0)
        assert api_manager.get_total_prompt_tokens() == 60
        assert api_manager.get_total_completion_tokens() == 120
        assert api_manager.get_total_cost() == (60 * 0.002 + 120 * 0.002) / 1000
        assert api_manager.get_total_budget() == 10.0

    @staticmethod
    def test_set_total_budget():
        """Test if setting the total budget works correctly."""
        total_budget = 10.0
        api_manager.set_total_budget(total_budget)

        assert api_manager.get_total_budget() == total_budget

    @staticmethod
    def test_update_cost():
        """Test if updating the cost works correctly."""
        prompt_tokens = 50
        completion_tokens = 100
        model = "gpt-3.5-turbo"

        api_manager.update_cost(prompt_tokens, completion_tokens, model)

        assert api_manager.get_total_prompt_tokens() == 50
        assert api_manager.get_total_completion_tokens() == 100
        assert api_manager.get_total_cost() == (50 * 0.002 + 100 * 0.002) / 1000
