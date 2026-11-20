"""A workflow on a model the catalog has switched off runs, and pays for, the
model's catalog fallback from the model's shutdown date.

Run the way a user's run goes, like ``agent_run_billing_test``: the graph is
saved and started through the API's handlers, and an executor pod runs it
against a fake Anthropic endpoint on localhost, on a day the pod's own script
sets. Sonnet 4.5 is switched off, with Sonnet 5.5 as its fallback from
2026-11-30, the day Anthropic retires it.

Each case starts an executor pod, which takes most of a minute, so this file
keeps one. ``backend/blocks/retired_model_fallback_test.py`` covers, without a
pod, a model wired in from a workflow input and the days before a shutdown date.
"""

from datetime import date
from uuid import uuid4

import pytest

from backend.api.model import CreateGraph
from backend.blocks.llm import AITextGeneratorBlock, LLMModel
from backend.data import graph as graph_model
from backend.data.block_cost_config import TOKEN_COST
from backend.data.credit import UserCredit
from backend.data.llm_registry import llm_models
from backend.data.user import get_or_create_user
from backend.integrations import credentials_store
from backend.util.test import SpinTestServer

from .agent_run_billing_test import (
    EXECUTOR_POD,
    PLATFORM_ANTHROPIC_KEY,
    _run_on_an_executor_pod,
    _start_run,
)
from .provider import INPUT_TOKENS_PER_REPLY, FakeAnthropic

SWITCHED_OFF = LLMModel.CLAUDE_4_5_SONNET
FALLBACK = LLMModel.CLAUDE_5_5_SONNET
SHUTDOWN = date(2026, 11, 30)


def _executor_pod_on(day: date) -> str:
    """The executor pod's script, with the day a shutdown date is read
    against set to *day*."""
    today = f"datetime.date({day.year}, {day.month}, {day.day})"
    return (
        "import datetime\n"
        "import backend.data.llm_registry.llm_models as llm_models\n"
        f"llm_models._today = lambda: {today}\n" + EXECUTOR_POD
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_workflow_on_a_switched_off_model_runs_and_pays_for_its_fallback(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    # The test process saves and starts the run, so it reads the same day as
    # the pod, not the calendar: the test must not change meaning after 11-30.
    monkeypatch.setattr(llm_models, "_today", lambda: SHUTDOWN)
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
        CreateGraph(graph=_one_model_call()), user.id
    )
    balance_before = await wallet.get_credits(user.id)

    async with FakeAnthropic(tool_calls=0, tool_calls_fail=False) as model:
        run_entry = await _start_run(graph, user.id, monkeypatch)
        await _run_on_an_executor_pod(run_entry, model.url, _executor_pod_on(SHUTDOWN))

    paid = balance_before - await wallet.get_credits(user.id)

    # Sonnet 5.5 answered the one call, and the wallet paid its price for it:
    # 300 credits, not Sonnet 4.5's 450.
    assert model.models == [FALLBACK.value]
    assert paid == _price_of_one_reply(FALLBACK)
    assert _price_of_one_reply(FALLBACK) < _price_of_one_reply(SWITCHED_OFF)


def _price_of_one_reply(model: LLMModel) -> int:
    return int(TOKEN_COST[model].input * INPUT_TOKENS_PER_REPLY / 1_000_000)


def _one_model_call() -> graph_model.Graph:
    platform_key = {
        "id": credentials_store.anthropic_credentials.id,
        "provider": "anthropic",
        "type": "api_key",
        "title": credentials_store.anthropic_credentials.title,
    }
    call = graph_model.Node(
        block_id=AITextGeneratorBlock().id,
        input_default={
            "prompt": "Say hello.",
            "model": SWITCHED_OFF.value,
            "credentials": platform_key,
        },
    )
    return graph_model.Graph(
        name="One model call", description="Calls the model once.", nodes=[call]
    )
