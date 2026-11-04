from unittest.mock import AsyncMock

import pytest

from backend.copilot.tasks_rubric import (
    TASK_CONTRACT,
    TASK_RUBRIC,
    get_tasks_supplement,
)


@pytest.mark.asyncio
async def test_the_rubric_reaches_accounts_with_tasks_on(monkeypatch):
    monkeypatch.setattr(
        "backend.copilot.tasks_rubric.is_feature_enabled",
        AsyncMock(return_value=True),
    )
    assert await get_tasks_supplement("user-1") == TASK_RUBRIC + TASK_CONTRACT


@pytest.mark.asyncio
async def test_accounts_without_tasks_are_not_told_about_task_tools(monkeypatch):
    monkeypatch.setattr(
        "backend.copilot.tasks_rubric.is_feature_enabled",
        AsyncMock(return_value=False),
    )
    assert await get_tasks_supplement("user-1") == ""
    assert await get_tasks_supplement(None) == ""


@pytest.mark.parametrize(
    "rule",
    [
        "isn't a task",
        "about two minutes and a few tool calls",
        "updates that task",
        "never duplicate it",
        "is a routine, not a task",
        "Repeating the same step several times stays one task",
        "never call `create_task` for a hire",
        "Take the task title from the user's own words",
        "mention only the final plan",
        "with Rename and Delete",
        "Sub-tasks are only for clearly different pieces of work",
        "make it a task even if it is small",
        "Create the task before any of its work",
        "not `delegate_to_expert`",
        "whoever works it asks with `ask_user`",
        "ask one question",
    ],
)
def test_every_rule_is_in_the_prompt(rule):
    assert rule in TASK_RUBRIC


@pytest.mark.parametrize(
    "rule",
    [
        "a quick task can have one check and no plan",
        "never ask the user to fill it in",
        "becomes the blocker on its own",
        "Never for routine progress",
        "One question at a time",
        "stays one step with `rows`",
        "your default and a one-line why",
    ],
)
def test_the_contract_rules_are_in_the_prompt(rule):
    assert rule in TASK_CONTRACT
