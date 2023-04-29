"""
Tests the ai_config class and how it handles configuration loading and command line overrides
"""

import pytest

from autogpt.config.ai_config import AIConfig


def test_command_line_overrides():
    """
    Test command line overrides for AI parameters are set correctly in the AI config.
    """
    ai_config = AIConfig.load(
        ai_name="testGPT", ai_role="testRole", ai_goals=["testGoal"]
    )

    assert ai_config.ai_name == "testGPT"
    assert ai_config.ai_role == "testRole"
    assert ai_config.ai_goals == ["testGoal"]


def test_command_line_overrides_with_config_file():
    """
    Test command line overrides for AI parameters are set correctly in the AI config.
    """
    ai_config = AIConfig.load(
        ai_name="testGPTOverride",
        ai_role="testRoleOverride",
        ai_goals=["testGoalOverride"],
        config_file="tests/test_config.yaml",
    )

    # Should have loaded from overrides and not from config
    assert ai_config.ai_name == "testGPTOverride"
    assert ai_config.ai_role == "testRoleOverride"
    assert ai_config.ai_goals == ["testGoalOverride"]


def test_command_line_override_singular():
    """
    Test we can supply one override and prompt for the rest
    """

    ai_config = AIConfig.load(ai_name="testGPTOverride")

    assert ai_config.ai_name == "testGPTOverride"
    assert ai_config.ai_role == None
    assert ai_config.ai_goals == []
