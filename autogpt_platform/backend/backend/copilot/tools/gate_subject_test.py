"""The gate decides a block or workflow run on what it runs, not on the tool.

Driven through ``BaseTool.execute`` so the subject hook, the gate and the
approval handed to the run are the ones the engines call.
"""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.blocks._base import BlockEffect
from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.ai_image_generator_block import AIImageGeneratorBlock
from backend.blocks.basic import StoreValueBlock
from backend.blocks.code_executor import ExecuteCodeStepBlock
from backend.blocks.discord.bot_blocks import SendDiscordMessageBlock
from backend.blocks.generic_webhook.triggers import GenericWebhookTriggerBlock
from backend.blocks.github.issues import GithubAddLabelBlock
from backend.blocks.github.repo_files import GithubCreateFileBlock
from backend.blocks.google.gmail import GmailSendBlock
from backend.blocks.http import SendWebRequestBlock
from backend.blocks.io import AgentInputBlock, AgentOutputBlock
from backend.blocks.search import GetWikipediaSummaryBlock
from backend.blocks.sql_query_block import SQLQueryBlock
from backend.copilot.gate.effects import block_effect, graph_effect
from backend.copilot.gate.policy import Effect
from backend.copilot.gate.subject import workflow_subject
from backend.copilot.model import AutopilotMode, ChatSession, ChatSessionMetadata
from backend.copilot.tools.models import BlockOutputResponse, ErrorResponse
from backend.copilot.tools.run_agent import RunAgentTool
from backend.copilot.tools.run_capability import RunCapabilityTool
from backend.data.graph import BaseGraph, GraphModel, Link, Node, NodeModel

_GATE = "backend.copilot.gate"
_CAP = "backend.copilot.tools.run_capability"
_AGENT_GRAPH = "backend.copilot.tools.run_agent._agent_graph"


def _session(mode: AutopilotMode = "auto") -> ChatSession:
    return ChatSession(
        session_id="session-1",
        user_id="user-1",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata=ChatSessionMetadata(origin="interactive", autopilot_mode=mode),
        messages=[],
    )


@pytest.fixture
def gate():
    """The gate on, no approval on record, nothing rejected in this chat."""
    store = SimpleNamespace(
        find_review=AsyncMock(return_value=None),
        open_review=AsyncMock(return_value=True),
        consume=AsyncMock(return_value=True),
    )
    with (
        patch(f"{_GATE}.is_feature_enabled", AsyncMock(return_value=True)),
        patch(f"{_GATE}.review_store.find_review", store.find_review),
        patch(f"{_GATE}.review_store.open_review", store.open_review),
        patch(f"{_GATE}.review_store.consume", store.consume),
        patch(f"{_GATE}.held.remember", AsyncMock(return_value=True)),
        patch(f"{_GATE}.chat_rules.ask_reason", AsyncMock(return_value=None)),
        patch(f"{_GATE}.chat_rules.set_ask", AsyncMock()),
        patch(f"{_GATE}.reads.release_held_read", AsyncMock(return_value=None)),
        patch(f"{_GATE}.reads.screen_read", AsyncMock(return_value=None)),
    ):
        yield store


@pytest.fixture
def ran():
    """What reached the block run, if anything did."""
    run = AsyncMock(
        return_value=BlockOutputResponse(
            message="ran",
            block_id="b",
            block_name="b",
            outputs={},
            success=True,
            session_id="session-1",
        )
    )
    with patch(f"{_CAP}._run_block", run):
        yield run


async def _run_capability(
    session: ChatSession, block_id: str, payload: dict[str, Any], **extra: Any
):
    return await RunCapabilityTool().execute(
        "user-1", session, "call-1", id=block_id, input=payload, **extra
    )


def _is_held(result) -> bool:
    return "approval_required" in str(result.output)


# ---- blocks through run_capability -----------------------------------------


@pytest.mark.parametrize("mode", ["ask_first", "auto"])
async def test_a_read_block_runs_without_a_card(gate, ran, mode):
    result = await _run_capability(
        _session(mode), GetWikipediaSummaryBlock().id, {"topic": "Otters"}
    )
    assert not _is_held(result)
    ran.assert_awaited_once()
    gate.open_review.assert_not_awaited()


@pytest.mark.parametrize("mode", ["ask_first", "auto"])
async def test_a_workspace_block_runs_without_a_card(gate, ran, mode):
    result = await _run_capability(
        _session(mode), AIImageGeneratorBlock().id, {"prompt": "an otter"}
    )
    assert not _is_held(result)
    ran.assert_awaited_once()


@pytest.mark.parametrize(
    "payload, extra",
    [
        ({}, {}),
        ({"channel_name": "general", "message_content": "hi"}, {"validate_only": True}),
    ],
    ids=["schema lookup", "validate_only"],
)
async def test_a_call_that_runs_nothing_never_asks(gate, ran, payload, extra):
    result = await _run_capability(
        _session("ask_first"), SendDiscordMessageBlock().id, payload, **extra
    )
    assert not _is_held(result)
    gate.open_review.assert_not_awaited()


async def test_an_external_block_asks_and_the_card_names_it(gate, ran):
    result = await _run_capability(
        _session(),
        SendDiscordMessageBlock().id,
        {"channel_name": "general", "message_content": "hi"},
    )
    assert _is_held(result)
    ran.assert_not_awaited()
    args = gate.open_review.await_args.args
    reason, subject = args[5], args[6]
    assert subject.name == "Send Discord Message"
    assert subject.key == f"block:{SendDiscordMessageBlock().id}"
    assert reason == "Runs Send Discord Message, which reaches outside the platform."
    assert subject.irreversible


async def test_the_chain_row_names_the_block_the_card_names(gate, ran):
    result = await _run_capability(
        _session(),
        SendDiscordMessageBlock().id,
        {"channel_name": "general", "message_content": "hi"},
    )
    output = json.loads(result.output)
    assert (output["ask"], output["object"]) == ("Run", "Send Discord Message")


async def test_an_unclassified_block_asks(gate, ran):
    result = await _run_capability(
        _session(),
        GithubAddLabelBlock().id,
        {"issue_url": "https://github.com/o/r/issues/1", "label": "bug"},
    )
    assert _is_held(result)
    assert (
        gate.open_review.await_args.args[5]
        == "Otto does not know what Github Add Label does, so he asks."
    )


async def test_an_approved_irreversible_block_asks_once(gate):
    """The card was the question; the irreversible-action pause must not be a second."""
    gate.find_review.return_value = SimpleNamespace(
        status=ReviewStatus.APPROVED, payload={}
    )
    prep = MagicMock(
        synthetic_node_id="node",
        input_data={"channel_name": "general", "message_content": "hi"},
        required_non_credential_keys=set(),
        provided_input_keys=set(),
    )
    hitl = AsyncMock(return_value=("exec-id", prep.input_data))
    execute = AsyncMock(return_value=MagicMock(type="block_output"))
    with (
        patch(
            "backend.copilot.tools.run_block.prepare_block_for_execution",
            AsyncMock(return_value=prep),
        ),
        patch(
            "backend.copilot.tools.run_block.get_current_permissions",
            return_value=None,
        ),
        patch(
            "backend.copilot.tools.run_block.check_spend_approval",
            AsyncMock(return_value=None),
        ),
        patch("backend.copilot.tools.run_block.check_hitl_review", hitl),
        patch("backend.copilot.tools.run_block.execute_block", execute),
    ):
        await _run_capability(_session(), SendDiscordMessageBlock().id, prep.input_data)
    hitl.assert_not_awaited()
    execute.assert_awaited_once()
    gate.open_review.assert_not_awaited()


async def test_the_model_cannot_forge_an_approval(gate, ran):
    await _run_capability(
        _session("unsupervised"),
        SendDiscordMessageBlock().id,
        {"channel_name": "general", "message_content": "hi"},
        _gate_approved=True,
    )
    assert ran.await_args.args[5] is False


async def test_a_rejection_asks_for_the_subject_not_the_tool(gate, ran):
    gate.find_review.return_value = SimpleNamespace(
        status=ReviewStatus.REJECTED, payload={}
    )
    call = SimpleNamespace(rule_key="block:abc")
    with (
        patch(f"{_GATE}.chat_rules.set_ask", AsyncMock()) as set_ask,
        patch(f"{_GATE}.held._held", AsyncMock(return_value={"x": call})),
        patch(f"{_GATE}.review_store.review_id_for", return_value="x"),
    ):
        await _run_capability(_session(), GetWikipediaSummaryBlock().id, {"topic": "x"})
    set_ask.assert_awaited_once_with("session-1", "block:abc")


async def test_the_gate_off_resolves_no_subject():
    """Flag off: no subject lookup, so no extra graph or registry work."""
    subject = AsyncMock()
    with (
        patch(f"{_GATE}.is_feature_enabled", AsyncMock(return_value=False)),
        patch.object(RunAgentTool, "gate_subject", subject),
        patch.object(RunAgentTool, "_execute", AsyncMock(return_value=_answer())),
        patch(f"{_GATE}.reads.screen_read", AsyncMock(return_value=None)),
        patch(f"{_GATE}.reads.release_held_read", AsyncMock(return_value=None)),
    ):
        await RunAgentTool().execute(
            "user-1", _session(), "call-1", library_agent_id="lib-1"
        )
    subject.assert_not_awaited()


# ---- workflows --------------------------------------------------------------


def test_a_read_workflow_with_structure_and_a_nested_run_reads():
    graph = _graph(
        [
            _node("in", AgentInputBlock(), {"name": "topic"}),
            _node("nested", AgentExecutorBlock(), {"graph_id": "sub"}),
            _node("read", GetWikipediaSummaryBlock(), {}),
            _node("out", AgentOutputBlock(), {"name": "summary"}),
        ],
        sub_graphs=[_sub("sub", [_node("s1", StoreValueBlock(), {})])],
    )
    assert workflow_subject(graph).effect is Effect.READ


def test_a_write_inside_a_sub_graph_names_it():
    graph = _graph(
        [
            _node("read", GetWikipediaSummaryBlock(), {}),
            _node("nested", AgentExecutorBlock(), {"graph_id": "sub"}),
        ],
        sub_graphs=[_sub("sub", [_node("send", GmailSendBlock(), {})])],
    )
    subject = workflow_subject(graph)
    assert subject.effect is Effect.EXTERNAL
    assert (
        subject.reason
        == "Runs Morning digest; its step Gmail Send reaches outside the platform."
    )
    assert subject.name == "Morning digest"


def test_a_nested_run_whose_sub_graph_is_missing_is_unreadable():
    graph = _graph([_node("nested", AgentExecutorBlock(), {"graph_id": "gone"})])
    assert graph_effect(graph).effect is None


@pytest.mark.parametrize(
    "linked, effect", [(False, Effect.READ), (True, Effect.EXTERNAL)]
)
def test_a_linked_method_makes_a_web_request_unreadable(linked, effect):
    links = [_link("in", "result", "req", "method")] if linked else []
    graph = _graph(
        [
            _node("in", AgentInputBlock(), {"name": "method"}),
            _node("req", SendWebRequestBlock(), {"url": "u", "method": "GET"}),
        ],
        links=links,
    )
    subject = workflow_subject(graph)
    assert subject.effect is effect
    assert subject.reason == (
        "Otto does not know what Send Web Request does, so he asks." if linked else ""
    )


def test_a_schedule_of_a_read_workflow_is_a_platform_edit():
    graph = _graph([_node("read", GetWikipediaSummaryBlock(), {})])
    assert workflow_subject(graph, schedules=True).effect is Effect.PLATFORM


async def test_a_trigger_workflow_shows_no_card(gate):
    graph = _graph(
        [
            _node("hook", GenericWebhookTriggerBlock(), {}),
            _node("send", GmailSendBlock(), {}),
        ]
    )
    details = _answer()
    with (
        patch(_AGENT_GRAPH, AsyncMock(return_value=(graph, None))),
        patch.object(RunAgentTool, "_execute", AsyncMock(return_value=details)),
    ):
        result = await RunAgentTool().execute(
            "user-1", _session(), "call-1", library_agent_id="lib-1"
        )
    assert not _is_held(result)
    gate.open_review.assert_not_awaited()


async def test_an_external_workflow_asks_naming_it(gate):
    graph = _graph([_node("send", GmailSendBlock(), {})])
    run = AsyncMock(return_value=_answer())
    with (
        patch(_AGENT_GRAPH, AsyncMock(return_value=(graph, None))),
        patch.object(RunAgentTool, "_execute", run),
    ):
        result = await RunAgentTool().execute(
            "user-1", _session(), "call-1", library_agent_id="lib-1"
        )
    assert _is_held(result)
    run.assert_not_awaited()
    assert gate.open_review.await_args.args[6].name == "Morning digest"


@pytest.mark.parametrize("approved", [True, False])
async def test_only_an_approved_run_skips_the_irreversible_pause(gate, approved):
    if approved:
        gate.find_review.return_value = SimpleNamespace(
            status=ReviewStatus.APPROVED, payload={}
        )
    graph = _graph([_node("send", GmailSendBlock(), {})])
    run = AsyncMock(return_value=_answer())
    with (
        patch(_AGENT_GRAPH, AsyncMock(return_value=(graph, MagicMock(id="lib-1")))),
        patch.object(RunAgentTool, "_run_agent", run),
        patch.object(
            RunAgentTool, "_check_prerequisites", AsyncMock(return_value=({}, None))
        ),
        patch(
            "backend.copilot.tools.run_agent.require_installed_workflow",
            AsyncMock(return_value=None),
        ),
    ):
        await RunAgentTool().execute(
            "user-1", _session("unsupervised"), "call-1", library_agent_id="lib-1"
        )
    assert run.await_args.kwargs["gate_approved"] is approved


@pytest.mark.parametrize("unreadable_first", [True, False])
@pytest.mark.parametrize("kind", ["undeclared block", "linked web request"])
def test_an_unreadable_node_never_hides_an_irreversible_one(unreadable_first, kind):
    """Approving the card lifts the pause, so the card must show the send."""
    unreadable = (
        _node("label", GithubAddLabelBlock(), {})
        if kind == "undeclared block"
        else _node("req", SendWebRequestBlock(), {"url": "u", "method": "GET"})
    )
    send = _node("send", GmailSendBlock(), {})
    nodes = [unreadable, send] if unreadable_first else [send, unreadable]
    links = [_link("in", "result", "req", "method")]
    subject = workflow_subject(
        _graph([_node("in", AgentInputBlock(), {"name": "m"}), *nodes], links=links)
    )
    assert subject.effect is Effect.EXTERNAL
    assert (
        subject.reason
        == "Runs Morning digest; its step Gmail Send reaches outside the platform."
    )


@pytest.mark.parametrize("send_first", [True, False])
def test_an_irreversible_node_names_an_external_workflow(send_first):
    nodes = [
        _node("send", GmailSendBlock(), {}),
        _node("file", GithubCreateFileBlock(), {}),
    ]
    subject = workflow_subject(_graph(nodes if send_first else nodes[::-1]))
    assert (
        subject.reason
        == "Runs Morning digest; its step Gmail Send reaches outside the platform."
    )


@pytest.mark.parametrize(
    "block, inputs, linked, effect",
    [
        (SendWebRequestBlock(), {}, (), BlockEffect.EXTERNAL),
        (SendWebRequestBlock(), {"method": "POST"}, (), BlockEffect.EXTERNAL),
        (SendWebRequestBlock(), {"method": "HEAD"}, (), BlockEffect.READ),
        (SQLQueryBlock(), {}, (), BlockEffect.READ),
        (SQLQueryBlock(), {"read_only": False}, (), BlockEffect.EXTERNAL),
        (SQLQueryBlock(), {"read_only": True}, ("read_only",), None),
    ],
    ids=["no method", "POST", "HEAD", "SQL default", "SQL write", "SQL linked"],
)
def test_input_decided_blocks_follow_their_input(block, inputs, linked, effect):
    assert block_effect(block, inputs, linked) is effect


async def test_a_workflow_the_lookup_misses_asks(gate):
    """The tool's own effect is external, so a miss asks rather than runs."""
    run = AsyncMock(return_value=_answer())
    with (
        patch(_AGENT_GRAPH, AsyncMock(return_value=(None, None))),
        patch.object(RunAgentTool, "_execute", run),
    ):
        result = await RunAgentTool().execute(
            "user-1", _session(), "call-1", library_agent_id="lib-1"
        )
    assert _is_held(result)
    run.assert_not_awaited()


async def test_saving_a_preset_of_a_read_workflow_is_a_platform_edit(gate):
    graph = _graph([_node("read", GetWikipediaSummaryBlock(), {})])
    with (
        patch(_AGENT_GRAPH, AsyncMock(return_value=(graph, None))),
        patch.object(RunAgentTool, "_execute", AsyncMock(return_value=_answer())),
    ):
        result = await RunAgentTool().execute(
            "user-1",
            _session("ask_first"),
            "call-1",
            library_agent_id="lib-1",
            save_as_preset=True,
            preset_name="otters",
        )
    assert _is_held(result)
    assert (
        gate.open_review.await_args.args[5] == "Runs Morning digest and saves a preset."
    )


async def test_a_graph_only_block_is_never_carded(gate, ran):
    """``run_block`` refuses it and the registry does not list it, so nothing runs."""
    result = await _run_capability(
        _session(), AgentExecutorBlock().id, {"graph_id": "x"}
    )
    assert not _is_held(result)
    gate.open_review.assert_not_awaited()


def _answer() -> ErrorResponse:
    return ErrorResponse(message="answered", session_id="session-1")


def _graph(
    nodes: list[NodeModel],
    links: list[Link] | None = None,
    sub_graphs: list[BaseGraph] | None = None,
) -> GraphModel:
    return GraphModel(
        id="main",
        version=1,
        name="Morning digest",
        description="",
        user_id="user-1",
        created_at=datetime.now(UTC),
        nodes=nodes,
        links=links or [],
        sub_graphs=sub_graphs or [],
    )


def _sub(graph_id: str, nodes: list[NodeModel]) -> BaseGraph:
    return BaseGraph(
        id=graph_id,
        version=1,
        name=graph_id,
        description="",
        nodes=[
            Node(**node.model_dump(include={"id", "block_id", "input_default"}))
            for node in nodes
        ],
    )


def _node(node_id: str, block, input_default: dict[str, Any]) -> NodeModel:
    return NodeModel(
        id=node_id,
        block_id=block.id,
        input_default=input_default,
        graph_id="main",
        graph_version=1,
    )


def _link(source: str, source_name: str, sink: str, sink_name: str) -> Link:
    return Link(
        source_id=source, source_name=source_name, sink_id=sink, sink_name=sink_name
    )


async def test_a_preset_this_chat_cannot_use_raises_no_card(gate):
    """Another expert's preset, or a missing one: the run refuses, so asking is noise."""
    run = AsyncMock(return_value=_answer())
    with (
        patch(
            "backend.copilot.tools.run_agent._preset_graph",
            AsyncMock(return_value=(None, None)),
        ),
        patch.object(RunAgentTool, "_execute", run),
    ):
        await RunAgentTool().execute(
            "user-1", _session("ask_first"), "call-1", preset_id="p-1"
        )
    gate.open_review.assert_not_awaited()
    run.assert_awaited_once()


# Execute Code itself resolves to bash_exec; this one is reached as a block.
_CODE = {
    "sandbox_id": "sbx-1",
    "language": "python",
    "step_code": "print(open('invoices.csv').read())",
}


async def test_a_code_block_in_auto_goes_to_the_supervisor_and_runs_on_a_vouch(
    gate, ran
):
    """Its effect is the code the call carries, so the check reads that code."""
    classify = AsyncMock(return_value=(True, ""))
    with patch(f"{_GATE}.classify", classify):
        result = await _run_capability(
            _session("auto"), ExecuteCodeStepBlock().id, dict(_CODE)
        )
    assert not _is_held(result)
    ran.assert_awaited_once()
    assert (
        classify.await_args.kwargs["args"]["input"]["step_code"] == _CODE["step_code"]
    )


async def test_a_code_block_the_supervisor_cannot_vouch_for_asks_with_its_reason(
    gate, ran
):
    classify = AsyncMock(return_value=(False, "it reads a local file of invoices"))
    with patch(f"{_GATE}.classify", classify):
        result = await _run_capability(
            _session("auto"), ExecuteCodeStepBlock().id, dict(_CODE)
        )
    assert _is_held(result)
    ran.assert_not_awaited()
    args, kwargs = gate.open_review.await_args
    assert args[5] == "it reads a local file of invoices"
    assert kwargs["reason_kind"] == "supervisor"


async def test_a_code_block_in_ask_first_asks_without_the_supervisor(gate, ran):
    classify = AsyncMock(return_value=(True, ""))
    with patch(f"{_GATE}.classify", classify):
        result = await _run_capability(
            _session("ask_first"), ExecuteCodeStepBlock().id, dict(_CODE)
        )
    assert _is_held(result)
    classify.assert_not_awaited()


async def test_the_reason_names_otto_as_he_even_in_an_experts_chat(gate, ran):
    """Otto is the supervisor in every chat, an expert's included."""
    session = _session().model_copy(update={"expert_id": "maria"})
    await _run_capability(
        session,
        GithubAddLabelBlock().id,
        {"issue_url": "https://github.com/o/r/issues/1", "label": "bug"},
    )
    reason = gate.open_review.await_args.args[5]
    assert reason == "Otto does not know what Github Add Label does, so he asks."


def test_a_workflow_with_two_irreversible_steps_names_both():
    """Approving the run covers every step, so the card names every one it covers."""
    subject = workflow_subject(
        _graph(
            [
                _node("send", GmailSendBlock(), {}),
                _node("post", SendDiscordMessageBlock(), {}),
            ]
        )
    )
    assert subject.irreversible
    assert subject.reason == (
        "Runs Morning digest; its steps Gmail Send and Send Discord Message "
        "reach outside the platform."
    )


async def test_a_workflow_called_without_its_inputs_asks_nothing(gate):
    """The run answers with the inputs it needs, so a card first is noise."""
    graph = _graph(
        [
            _node("in", AgentInputBlock(), {"name": "topic"}),
            _node("send", GmailSendBlock(), {}),
        ]
    )
    run = AsyncMock(return_value=_answer())
    with (
        patch(_AGENT_GRAPH, AsyncMock(return_value=(graph, None))),
        patch.object(RunAgentTool, "_execute", run),
    ):
        await RunAgentTool().execute(
            "user-1", _session("ask_first"), "call-1", library_agent_id="lib-1"
        )
    gate.open_review.assert_not_awaited()
    run.assert_awaited_once()
