import pytest
from unittest.mock import Mock, patch
from autogpt.logs import print_assistant_thoughts, _remaining_budget_description


# Test when total_budget is greater than 0
def test_remaining_budget_description_positive_budget():
    with patch(
        "autogpt.api_manager.ApiManager.get_total_budget", return_value=100
    ), patch("autogpt.api_manager.ApiManager.get_total_cost", return_value=50):
        assert _remaining_budget_description() == "$50 remaining from $100."


# Test when total_budget is 0
def test_remaining_budget_description_zero_budget():
    with patch(
        "autogpt.api_manager.ApiManager.get_total_budget", return_value=0
    ), patch("autogpt.api_manager.ApiManager.get_total_cost", return_value=0):
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
        "autogpt.api_manager.ApiManager.get_total_budget", return_value=100
    ), patch("autogpt.api_manager.ApiManager.get_total_cost", return_value=50):
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
        "autogpt.api_manager.ApiManager.get_total_budget", return_value=0
    ), patch("autogpt.api_manager.ApiManager.get_total_cost", return_value=50):
        print_assistant_thoughts(
            "AI_NAME", assistant_reply_json_valid, speak_mode=False
        )

        out, err = capfd.readouterr()
        assert "BUDGET:" not in out
