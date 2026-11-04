import json

import pytest

from backend.blocks.basic import StoreValueBlock
from backend.blocks.generic_webhook.triggers import GenericWebhookTriggerBlock

from .examples import Verdict
from .runner import parse_call, turn_over
from .verdict import ToolCall, TurnVerdict, asked_needlessly, passes, turn_verdict


@pytest.mark.parametrize(
    "calls, verdict",
    [
        ([], Verdict.NOT_A_TASK),
        ([ToolCall(name="web_search")], Verdict.NOT_A_TASK),
        ([ToolCall(name="create_task")], Verdict.TASK),
        ([ToolCall(name="propose_plan")], Verdict.TASK),
        ([ToolCall(name="hire_expert")], Verdict.TASK),
        ([ToolCall(name="raise_expert")], Verdict.TASK),
        (
            [ToolCall(name="propose_plan", args={"split": True})],
            Verdict.PARENT_WITH_SUBTASKS,
        ),
        (
            [ToolCall(name="create_task", args={"parent_task_id": "task-18"})],
            Verdict.UPDATE_EXISTING,
        ),
        ([ToolCall(name="comment_on_task")], Verdict.UPDATE_EXISTING),
        ([ToolCall(name="update_task_status")], Verdict.UPDATE_EXISTING),
        ([ToolCall(name="schedule_routine")], Verdict.ROUTINE),
        ([ToolCall(name="schedule_followup")], Verdict.ROUTINE),
        ([ToolCall(name="run_agent", args={"cron": "0 9 * * 1"})], Verdict.ROUTINE),
        ([ToolCall(name="run_agent")], Verdict.NOT_A_TASK),
    ],
)
def test_a_tool_call_maps_to_its_outcome(calls, verdict):
    assert turn_verdict(calls).outcome == verdict


def test_the_first_committing_call_decides():
    calls = [
        ToolCall(name="list_my_tasks"),
        ToolCall(name="comment_on_task"),
        ToolCall(name="create_task"),
    ]

    assert turn_verdict(calls) == TurnVerdict(
        outcome=Verdict.UPDATE_EXISTING, asked_first=False
    )


def test_a_question_is_recorded_and_the_next_action_decides():
    calls = [ToolCall(name="ask_question"), ToolCall(name="create_task")]

    assert turn_verdict(calls) == TurnVerdict(outcome=Verdict.TASK, asked_first=True)


def test_ask_then_task_needs_both_the_question_and_the_task():
    asked_then_task = TurnVerdict(outcome=Verdict.TASK, asked_first=True)
    task_only = TurnVerdict(outcome=Verdict.TASK, asked_first=False)
    asked_only = TurnVerdict(outcome=Verdict.NOT_A_TASK, asked_first=True)

    assert passes(Verdict.ASK_THEN_TASK, asked_then_task)
    assert not passes(Verdict.ASK_THEN_TASK, task_only)
    assert not passes(Verdict.ASK_THEN_TASK, asked_only)
    assert not asked_needlessly(Verdict.ASK_THEN_TASK, asked_then_task)


def test_a_question_before_the_right_task_passes_but_counts_as_needless():
    got = TurnVerdict(outcome=Verdict.TASK, asked_first=True)

    assert passes(Verdict.TASK, got)
    assert asked_needlessly(Verdict.TASK, got)


def test_a_split_plan_the_tool_would_refuse_does_not_decide():
    one_step = json.dumps(
        {"goal": "Launch", "steps": [{"title": "Blog"}], "split": True}
    )
    two_steps = json.dumps(
        {
            "goal": "Launch",
            "steps": [{"title": "Blog"}, {"title": "Email"}],
            "split": True,
        }
    )

    refused = parse_call("propose_plan", one_step)
    accepted = parse_call("propose_plan", two_steps)

    assert turn_verdict([refused]).outcome == Verdict.NOT_A_TASK
    assert turn_verdict([accepted]).outcome == Verdict.PARENT_WITH_SUBTASKS


def test_a_call_through_run_capability_is_read_as_the_tool_it_runs():
    call = parse_call(
        "run_capability",
        json.dumps({"id": "tool:schedule_routine", "input": {"cron": "0 9 * * 1"}}),
    )

    assert call.name == "schedule_routine"
    assert turn_verdict([call]).outcome == Verdict.ROUTINE


def test_a_workflow_built_from_a_trigger_is_a_routine():
    trigger = {"nodes": [{"id": "t", "block_id": GenericWebhookTriggerBlock().id}]}
    plain = {"nodes": [{"id": "s", "block_id": StoreValueBlock().id}]}

    built = ToolCall(name="create_agent", args={"agent_json": trigger})
    scraper = ToolCall(name="create_agent", args={"agent_json": plain})

    assert turn_verdict([built]).outcome == Verdict.ROUTINE
    assert turn_verdict([scraper]).outcome == Verdict.NOT_A_TASK


def test_the_turn_ends_at_a_decision_or_a_second_question():
    assert not turn_over([ToolCall(name="ask_question")])
    assert turn_over([ToolCall(name="ask_question"), ToolCall(name="ask_question")])
    assert turn_over([ToolCall(name="create_task")])
