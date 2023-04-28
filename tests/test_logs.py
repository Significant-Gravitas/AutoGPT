from unittest.mock import Mock, patch

import pytest

from autogpt.logs import (
    _remaining_budget_description,
    print_assistant_thoughts,
    remove_color_codes,
)


@pytest.mark.parametrize(
    "raw_text, clean_text",
    [
        (
            "COMMAND = \x1b[36mbrowse_website\x1b[0m  ARGUMENTS = \x1b[36m{'url': 'https://www.google.com', 'question': 'What is the capital of France?'}\x1b[0m",
            "COMMAND = browse_website  ARGUMENTS = {'url': 'https://www.google.com', 'question': 'What is the capital of France?'}",
        ),
        (
            "{'Schaue dir meine Projekte auf github () an, als auch meine Webseiten': 'https://github.com/Significant-Gravitas/Auto-GPT, https://discord.gg/autogpt und https://twitter.com/SigGravitas'}",
            "{'Schaue dir meine Projekte auf github () an, als auch meine Webseiten': 'https://github.com/Significant-Gravitas/Auto-GPT, https://discord.gg/autogpt und https://twitter.com/SigGravitas'}",
        ),
        ("", ""),
        ("hello", "hello"),
        ("hello\x1B[31m world", "hello world"),
        ("\x1B[36mHello,\x1B[32m World!", "Hello, World!"),
        (
            "\x1B[1m\x1B[31mError:\x1B[0m\x1B[31m file not found",
            "Error: file not found",
        ),
    ],
)
def test_remove_color_codes(raw_text, clean_text):
    assert remove_color_codes(raw_text) == clean_text


# Test when total_budget is greater than 0
def test_remaining_budget_description_positive_budget():
    with patch(
        "autogpt.llm.api_manager.ApiManager.get_total_budget", return_value=100
    ), patch("autogpt.llm.api_manager.ApiManager.get_total_cost", return_value=50):
        assert _remaining_budget_description() == "$50 remaining from $100."


# Test when total_budget is 0
def test_remaining_budget_description_zero_budget():
    with patch(
        "autogpt.llm.api_manager.ApiManager.get_total_budget", return_value=0
    ), patch("autogpt.llm.api_manager.ApiManager.get_total_cost", return_value=0):
        assert _remaining_budget_description() == "$0 remaining from $0."


def test_print_assistant_thoughts_budget(capfd):
    assistant_reply_json_valid = {
        "thoughts": {
            "text": "Test text",
            "reasoning": "Test reasoning",
            "plan": "Test plan",
            "criticism": "Test criticism",
        },
    }

    with patch(
        "autogpt.llm.api_manager.ApiManager.get_total_budget", return_value=100
    ), patch("autogpt.llm.api_manager.ApiManager.get_total_cost", return_value=50):
        print_assistant_thoughts(
            "AI_NAME", assistant_reply_json_valid, speak_mode=False
        )

        out, err = capfd.readouterr()
        assert "BUDGET:" in out
        assert "$50 remaining from $100." in out


def test_print_assistant_thoughts_no_budget(capfd):
    assistant_reply_json_valid = {
        "thoughts": {
            "text": "Test text",
            "reasoning": "Test reasoning",
            "plan": "Test plan",
            "criticism": "Test criticism",
        },
    }

    with patch(
        "autogpt.llm.api_manager.ApiManager.get_total_budget", return_value=0
    ), patch("autogpt.llm.api_manager.ApiManager.get_total_cost", return_value=50):
        print_assistant_thoughts(
            "AI_NAME", assistant_reply_json_valid, speak_mode=False
        )

        out, err = capfd.readouterr()
        assert "BUDGET:" not in out
