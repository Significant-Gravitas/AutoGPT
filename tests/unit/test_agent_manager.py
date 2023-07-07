import pytest

from autogpt.llm import ChatModelResponse
from autogpt.llm.chat import create_chat_completion
from autogpt.llm.providers.openai import OPEN_AI_CHAT_MODELS


@pytest.fixture
def task():
    return "translate English to French"


@pytest.fixture
def prompt():
    return "Translate the following English text to French: 'Hello, how are you?'"


@pytest.fixture
def model():
    return "gpt-3.5-turbo"


@pytest.fixture(autouse=True)
def mock_create_chat_completion(mocker, config):
    mock_create_chat_completion = mocker.patch(
        "autogpt.agent.agent_manager.create_chat_completion",
        wraps=create_chat_completion,
    )
    mock_create_chat_completion.return_value = ChatModelResponse(
        model_info=OPEN_AI_CHAT_MODELS[config.fast_llm_model],
        content="irrelevant",
        function_call={},
    )
    return mock_create_chat_completion
