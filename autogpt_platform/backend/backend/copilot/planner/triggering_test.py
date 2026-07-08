"""Tests for the planner triggering heuristic."""

import pytest

from backend.copilot.planner.triggering import is_multi_step_request


class TestIsMultiStepRequest:
    @pytest.mark.parametrize("msg", [None, "", "hi", "thanks!", "what's up?"])
    def test_trivial_messages_do_not_trigger(self, msg):
        assert is_multi_step_request(msg) is False

    @pytest.mark.parametrize(
        "msg",
        [
            "Build me an agent that scrapes GitHub issues and emails a summary",
            "Create a workflow that fetches the weather then posts it to Slack",
            "Set up an automation to research competitors and generate a report",
            "First fetch the data, and then transform it, after that upload it",
        ],
    )
    def test_task_like_messages_trigger(self, msg):
        assert is_multi_step_request(msg) is True

    def test_long_but_conversational_does_not_trigger(self):
        # Clears the length gate but has no task keyword.
        assert (
            is_multi_step_request(
                "I was just wondering how your day has been going so far honestly"
            )
            is False
        )

    def test_short_with_keyword_below_length_gate_does_not_trigger(self):
        # 'build' present but too short to be worth an expensive planner call.
        assert is_multi_step_request("build it") is False
