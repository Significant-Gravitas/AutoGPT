import json

import pytest

from autogpt.config import Config
from autogpt.memory import get_memory
from autogpt.memory_management.store_memory import (
    save_memory_trimmed_from_context_window,
)
from tests.utils import requires_api_key


@pytest.fixture
def message_history_fixture():
    assistant_reply = {
        "thoughts": {
            "text": "thoughts",
            "reasoning": "reasoning",
            "plan": "plan",
            "criticism": "criticism",
            "speak": "speak",
        },
        "command": {"name": "google", "args": {"query": "google_query"}},
    }
    return [
        {"content": json.dumps(assistant_reply, indent=4)},
        {"content": "Command Result: Important Information."},
    ]


@pytest.fixture
def expected_permanent_memory() -> str:
    return """Assistant Reply: {
    "thoughts": {
        "text": "thoughts",
        "reasoning": "reasoning",
        "plan": "plan",
        "criticism": "criticism",
        "speak": "speak"
    },
    "command": {
        "name": "google",
        "args": {
            "query": "google_query"
        }
    }
}
Result: None
Human Feedback:Command Result: Important Information."""


@requires_api_key("OPENAI_API_KEY")
@pytest.mark.vcr
def test_save_memory_trimmed_from_context_window(
    message_history_fixture, expected_permanent_memory, config: Config
):
    next_message_to_add_index = len(message_history_fixture) - 1
    memory = get_memory(config, init=True)
    save_memory_trimmed_from_context_window(
        message_history_fixture, next_message_to_add_index, memory
    )

    memory_found = memory.get_relevant("Important Information", 5)
    assert memory_found[0] == expected_permanent_memory
