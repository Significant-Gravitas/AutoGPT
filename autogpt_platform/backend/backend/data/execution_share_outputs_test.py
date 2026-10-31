"""Shared-run outputs match the owner's view (#15652).

``get_graph_execution_by_share_token`` used to hand-roll output extraction
from ``executionData`` without loading ``Input`` rows, so an Agent Output
block whose value was typed in the builder (a starting node: ``Input`` rows,
no ``executionData``) never reached the public page, and moderation-cleared
values weren't applied.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import prisma.models
import pytest
from prisma.enums import AgentExecutionStatus

from backend.blocks.io import AgentInputBlock, AgentOutputBlock
from backend.data.execution import (
    GraphExecution,
    get_graph_execution,
    get_graph_execution_by_share_token,
)

OUTPUT_BLOCK_ID = AgentOutputBlock().id
INPUT_BLOCK_ID = AgentInputBlock().id
MODERATED = "Failed due to content moderation (Moderation ID: mod-1)"
_T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def _io(name: str, value: Any, n: int) -> dict:
    return {
        "id": f"io-{name}-{n}",
        "name": name,
        "data": json.dumps(value),
        "time": _T0,
    }


def _node_exec(
    exec_id: str,
    *,
    block_id: str = OUTPUT_BLOCK_ID,
    status: AgentExecutionStatus = AgentExecutionStatus.COMPLETED,
    queued_s: int,
    execution_data: dict | None = None,
    inputs: dict | None = None,
    stats: dict | None = None,
    outputs: list | None = None,
) -> dict:
    return {
        "id": exec_id,
        "agentGraphExecutionId": "ge-1",
        "agentNodeId": f"node-{exec_id}",
        "executionStatus": status,
        "addedTime": _T0,
        "queuedTime": _T0 + timedelta(seconds=queued_s),
        "executionData": (
            json.dumps(execution_data) if execution_data is not None else None
        ),
        "stats": json.dumps(stats) if stats is not None else None,
        "Input": [
            _io(name, value, i)
            for i, (name, value) in enumerate((inputs or {}).items())
        ],
        "Output": [_io("output", value, i) for i, value in enumerate(outputs or [])],
        "Node": {
            "id": f"node-{exec_id}",
            "agentBlockId": block_id,
            "agentGraphId": "g-1",
            "agentGraphVersion": 1,
            "constantInput": "{}",
            "metadata": "{}",
        },
    }


def _execution() -> prisma.models.AgentGraphExecution:
    return prisma.models.AgentGraphExecution.model_validate(
        {
            "id": "ge-1",
            "agentGraphId": "g-1",
            "agentGraphVersion": 1,
            "userId": "owner-1",
            "createdAt": _T0,
            "executionStatus": AgentExecutionStatus.COMPLETED,
            "isDeleted": False,
            "isShared": True,
            "shareToken": "tok-1",
            "visibility": "PRIVATE",
            "AgentGraph": {
                "id": "g-1",
                "version": 1,
                "name": "Shared graph",
                "userId": "owner-1",
                "createdAt": _T0,
                "isActive": True,
                "visibility": "PRIVATE",
            },
            "NodeExecutions": [
                # Linked Agent Output: executionData written when enqueued.
                _node_exec(
                    "linked",
                    queued_s=5,
                    execution_data={"name": "linked_note", "value": "from input"},
                ),
                # Static Agent Outputs typed in the builder: Input rows only.
                _node_exec(
                    "static-str",
                    queued_s=1,
                    inputs={"name": "static_note", "value": "typed in builder"},
                ),
                _node_exec(
                    "static-num",
                    queued_s=2,
                    inputs={"name": "static_number", "value": 42},
                ),
                # Two values for one output name: queue-time order, not row order.
                _node_exec(
                    "multi-late",
                    queued_s=9,
                    execution_data={"name": "multi", "value": "second"},
                ),
                _node_exec(
                    "multi-early",
                    queued_s=3,
                    execution_data={"name": "multi", "value": "first"},
                ),
                # Moderated output: original executionData is still on the row.
                _node_exec(
                    "moderated",
                    status=AgentExecutionStatus.FAILED,
                    queued_s=4,
                    execution_data={"name": "secret", "value": "original content"},
                    stats={
                        "error": MODERATED,
                        "cleared_inputs": {"name": [MODERATED], "value": [MODERATED]},
                        "cleared_outputs": {},
                    },
                ),
                # Not yet queued: owner view drops it.
                _node_exec(
                    "incomplete",
                    status=AgentExecutionStatus.INCOMPLETE,
                    queued_s=6,
                    execution_data={"name": "pending", "value": "not run"},
                ),
            ],
        }
    )


@pytest.fixture
def exec_client(mocker):
    client = mocker.AsyncMock()
    client.find_first.return_value = _execution()
    mocker.patch.object(
        prisma.models.AgentGraphExecution, "prisma", return_value=client
    )
    return client


@pytest.mark.asyncio
async def test_shared_outputs_equal_owner_outputs(exec_client):
    owner = await get_graph_execution("owner-1", "ge-1", include_node_executions=True)
    shared = await get_graph_execution_by_share_token("tok-1")

    assert owner is not None and shared is not None
    assert dict(shared.outputs) == dict(owner.outputs)
    assert dict(shared.outputs) == {
        "static_note": ["typed in builder"],
        "static_number": [42],
        "multi": ["first", "second"],
        MODERATED: [MODERATED],
        "linked_note": ["from input"],
    }
    assert shared.graph_name == "Shared graph"


@pytest.mark.asyncio
async def test_shared_outputs_hide_moderated_original_value(exec_client):
    shared = await get_graph_execution_by_share_token("tok-1")

    assert shared is not None
    assert "secret" not in shared.outputs
    assert "original content" not in json.dumps(shared.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_share_query_uses_owner_node_execution_window(exec_client):
    await get_graph_execution("owner-1", "ge-1", include_node_executions=True)
    owner_nodes = exec_client.find_first.call_args.kwargs["include"]["NodeExecutions"]
    await get_graph_execution_by_share_token("tok-1")

    include = exec_client.find_first.call_args.kwargs["include"]
    node_execs = include["NodeExecutions"]
    assert include["AgentGraph"] is True
    assert "Input" in node_execs["include"]
    assert node_execs == owner_nodes


@pytest.mark.asyncio
async def test_unknown_share_token_returns_none(exec_client):
    exec_client.find_first.return_value = None

    assert await get_graph_execution_by_share_token("nope") is None


# Run outputs come from what the Agent Output block yielded (#15905).


async def _yielded(**inputs: Any) -> list:
    block = AgentOutputBlock()
    return [
        value
        async for pin, value in block.run(AgentOutputBlock.Input(**inputs))
        if pin == "output"
    ]


async def _run_outputs(**inputs: Any) -> list:
    execution = _execution()
    execution.NodeExecutions = [
        prisma.models.AgentNodeExecution.model_validate(
            _node_exec(
                "out",
                queued_s=1,
                inputs=inputs,
                outputs=await _yielded(**inputs),
            )
        )
    ]
    return GraphExecution.from_db(execution).outputs[inputs["name"]]


@pytest.mark.asyncio
async def test_run_outputs_use_formatted_value():
    assert await _run_outputs(
        name="shout", value="hi", format="{{ shout | upper }}"
    ) == ["HI"]


@pytest.mark.asyncio
async def test_run_outputs_use_html_escaped_value():
    (value,) = await _run_outputs(
        name="note", value="<b>x</b>", format="{{ note }}", escape_html=True
    )
    assert "&lt;b&gt;x&lt;/b&gt;" in value
    assert "<b>" not in value


@pytest.mark.asyncio
async def test_run_outputs_show_format_error():
    (value,) = await _run_outputs(
        name="probe", value="raw", format="{{ missing.attr }}"
    )
    assert value.startswith("Error: ")
    assert value.endswith(", raw")


@pytest.mark.asyncio
async def test_run_outputs_without_format_are_unchanged():
    value = {"bed": "4B"}
    assert await _run_outputs(name="patient", value=value) == [value]
