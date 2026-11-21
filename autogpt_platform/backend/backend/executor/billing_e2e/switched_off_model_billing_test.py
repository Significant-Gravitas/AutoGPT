"""A workflow on a model the catalog has switched off runs, and pays for, the
model's catalog fallback from the model's shutdown date. A node that started
before midnight on that date runs, and pays for, the model itself to the end.

Run the way a user's run goes, like ``agent_run_billing_test``: the graph is
saved and started through the API's handlers, and an executor pod runs it
against a fake Anthropic endpoint on localhost. Sonnet 4.5 is switched off,
with Sonnet 5.5 as its fallback from 2026-11-30, the day Anthropic retires it.
The pod and the test read the day from one clock. It says 29 Nov until the
provider gets the first request, so midnight passes while the agent's first
call is in flight.

The agent runs in agent mode on Sonnet 4.5 and calls an AI block, also saved
on Sonnet 4.5, as its tool. The agent started on 29 Nov, so all its calls run
Sonnet 4.5 and are billed at its price, though it settles on 30 Nov. Each tool
call is a node the agent starts on 30 Nov, which runs, and pays for, Sonnet
5.5. So does the AI block the agent's answer goes to: a node the graph
executor's own dispatcher charges and starts on 30 Nov.

Each case starts an executor pod, which takes most of a minute, so this file
keeps one. Without a pod, ``backend/blocks/retired_model_fallback_test.py``
covers a model wired in from a workflow input, the days before a shutdown date
and a run paused for review, and
``backend/executor/manager_shutdown_midnight_billing_test.py`` covers a node
across the midnight on 11 Dec too, a charge and a start either side of it, a
stop, reruns and the cost log.
"""

import os
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest

from backend.api.model import CreateGraph
from backend.blocks.llm import AITextGeneratorBlock, LLMModel
from backend.data import graph as graph_model
from backend.data.block_cost_config import TOKEN_COST
from backend.data.credit import UserCredit
from backend.data.execution import GraphExecutionEntry, get_graph_execution
from backend.data.llm_registry import llm_models
from backend.data.user import get_or_create_user
from backend.integrations import credentials_store
from backend.util.test import SpinTestServer

from .agent_run_billing_test import (
    EXECUTOR_POD,
    PLATFORM_ANTHROPIC_KEY,
    TOOL_CALLS,
    _agent_with_an_ai_tool,
    _run_on_an_executor_pod,
    _start_run,
)
from .provider import INPUT_TOKENS_PER_REPLY, FakeAnthropic

SWITCHED_OFF = LLMModel.CLAUDE_4_5_SONNET
FALLBACK = LLMModel.CLAUDE_5_5_SONNET
SHUTDOWN = date(2026, 11, 30)
THE_DAY_BEFORE = date(2026, 11, 29)


def _set_clock(clock: Path, day: date) -> None:
    """Point the clock at *day* in one step, so the pod never reads it half
    written."""
    pending = clock.with_suffix(".next")
    pending.write_text(day.isoformat())
    os.replace(pending, clock)


def _executor_pod_reading(clock: Path) -> str:
    """The executor pod's script, reading the day a shutdown date is read
    against from the file *clock*."""
    return (
        "import datetime, pathlib\n"
        "import backend.data.llm_registry.llm_models as llm_models\n"
        f"_clock = pathlib.Path({str(clock)!r})\n"
        "llm_models._today = lambda: datetime.date.fromisoformat(_clock.read_text())\n"
        + EXECUTOR_POD
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_an_agent_running_across_the_shutdown_midnight_pays_for_the_models_it_ran(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    clock = tmp_path / "today"
    _set_clock(clock, THE_DAY_BEFORE)
    # The test process saves and starts the run on the pod's clock, not the
    # calendar: the test must not change meaning after 11-30.
    monkeypatch.setattr(
        llm_models, "_today", lambda: date.fromisoformat(clock.read_text())
    )
    monkeypatch.setattr(
        credentials_store.settings.secrets, "anthropic_api_key", PLATFORM_ANTHROPIC_KEY
    )
    user = await get_or_create_user(
        {
            "sub": str(uuid4()),
            "email": f"billing-e2e-{uuid4()}@example.com",
            "name": "Billing E2E",
        }
    )
    wallet = UserCredit()
    await wallet.grant_credits(user.id, 10_000, "billing e2e")
    graph = await server.agent_server.test_create_graph(
        CreateGraph(graph=_agent_then_an_ai_block(SWITCHED_OFF)), user.id
    )
    balance_before = await wallet.get_credits(user.id)

    async with FakeAnthropic(
        tool_calls=TOOL_CALLS,
        tool_calls_fail=False,
        on_request=lambda: _set_clock(clock, SHUTDOWN),
    ) as model:
        run_entry = await _start_run(graph, user.id, monkeypatch)
        await _run_on_an_executor_pod(
            run_entry, model.url, _executor_pod_reading(clock)
        )

    paid = balance_before - await wallet.get_credits(user.id)
    run = await get_graph_execution(
        user.id, GraphExecutionEntry.model_validate_json(run_entry).graph_exec_id
    )

    # The agent ran the model it started on for each of its calls. Each tool
    # call, a node started after midnight, ran the fallback, and so did the
    # AI block the agent's answer went to, last.
    agent, fallback = SWITCHED_OFF.value, FALLBACK.value
    assert model.models == [agent, fallback, agent, fallback, agent, fallback]
    # The wallet paid each model's price for its calls: 3 x 450 + 3 x 300
    # credits, not 6 x 300. The run page shows what the wallet paid.
    agent_calls = TOOL_CALLS + 1
    assert paid == agent_calls * _price_of_one_reply(SWITCHED_OFF) + (
        TOOL_CALLS + 1
    ) * _price_of_one_reply(FALLBACK)
    assert run is not None and run.stats is not None
    assert run.stats.cost == paid


def _agent_then_an_ai_block(model: LLMModel) -> graph_model.Graph:
    """The agent with an AI tool, both on *model*, and an AI block on
    *model* that the agent's answer goes to."""
    graph = _agent_with_an_ai_tool(model)
    agent = next(
        node for node in graph.nodes if node.block_id != AITextGeneratorBlock().id
    )
    reply = graph_model.Node(
        block_id=AITextGeneratorBlock().id,
        input_default={
            "model": model.value,
            "credentials": agent.input_default["credentials"],
        },
    )
    graph.nodes.append(reply)
    graph.links.append(
        graph_model.Link(
            source_id=agent.id,
            sink_id=reply.id,
            source_name="finished",
            sink_name="prompt",
        )
    )
    return graph


def _price_of_one_reply(model: LLMModel) -> int:
    return int(TOKEN_COST[model].input * INPUT_TOKENS_PER_REPLY / 1_000_000)
