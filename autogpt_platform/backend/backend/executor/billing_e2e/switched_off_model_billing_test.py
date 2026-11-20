"""A workflow on a model the catalog has switched off runs, and pays for, the
model's catalog fallback.

Run the way a user's run goes, like ``agent_run_billing_test``: the graph is
saved and started through the API's handlers, and an executor pod runs it
against a fake Anthropic endpoint on localhost. Sonnet 4.5 is switched off,
with Sonnet 5.5 as its fallback. The model reaches the block from the node's
saved input, or from a workflow input wired into its ``model`` pin, a value no
pass over stored graphs rewrites.
"""

from typing import Literal
from uuid import uuid4

import pytest

from backend.api.model import CreateGraph
from backend.blocks.io import AgentInputBlock
from backend.blocks.llm import AITextGeneratorBlock, LLMModel
from backend.data import graph as graph_model
from backend.data.block_cost_config import TOKEN_COST
from backend.data.credit import UserCredit
from backend.data.user import get_or_create_user
from backend.integrations import credentials_store
from backend.util.test import SpinTestServer

from .agent_run_billing_test import (
    PLATFORM_ANTHROPIC_KEY,
    _run_on_an_executor_pod,
    _start_run,
)
from .provider import INPUT_TOKENS_PER_REPLY, FakeAnthropic

SWITCHED_OFF = LLMModel.CLAUDE_4_5_SONNET
FALLBACK = LLMModel.CLAUDE_5_5_SONNET


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("model_from", ["saved-input", "workflow-input"])
async def test_a_workflow_on_a_switched_off_model_runs_and_pays_for_its_fallback(
    server: SpinTestServer,
    monkeypatch: pytest.MonkeyPatch,
    model_from: Literal["saved-input", "workflow-input"],
):
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
        CreateGraph(graph=_one_model_call(model_from)), user.id
    )
    balance_before = await wallet.get_credits(user.id)

    async with FakeAnthropic(tool_calls=0, tool_calls_fail=False) as model:
        run_entry = await _start_run(
            graph,
            user.id,
            monkeypatch,
            inputs=(
                {"model": SWITCHED_OFF.value}
                if model_from == "workflow-input"
                else None
            ),
        )
        await _run_on_an_executor_pod(run_entry, model.url)

    paid = balance_before - await wallet.get_credits(user.id)

    # Sonnet 5.5 answered the one call, and the wallet paid Sonnet 5.5's price
    # for it: 300 credits today, where Sonnet 4.5's is 450.
    assert model.models == [FALLBACK.value]
    assert paid == _price_of_one_reply(FALLBACK) < _price_of_one_reply(SWITCHED_OFF)


def _price_of_one_reply(model: LLMModel) -> int:
    return int(TOKEN_COST[model].input * INPUT_TOKENS_PER_REPLY / 1_000_000)


def _one_model_call(
    model_from: Literal["saved-input", "workflow-input"],
) -> graph_model.Graph:
    platform_key = {
        "id": credentials_store.anthropic_credentials.id,
        "provider": "anthropic",
        "type": "api_key",
        "title": credentials_store.anthropic_credentials.title,
    }
    call = graph_model.Node(
        block_id=AITextGeneratorBlock().id,
        input_default={"prompt": "Say hello.", "credentials": platform_key},
    )
    if model_from == "saved-input":
        call.input_default["model"] = SWITCHED_OFF.value
        return graph_model.Graph(
            name="One model call", description="Calls the model once.", nodes=[call]
        )
    model_input = graph_model.Node(
        block_id=AgentInputBlock().id,
        input_default={"name": "model", "title": "Model"},
    )
    return graph_model.Graph(
        name="One model call",
        description="Calls the model the run names, once.",
        nodes=[model_input, call],
        links=[
            graph_model.Link(
                source_id=model_input.id,
                sink_id=call.id,
                source_name="result",
                sink_name="model",
            )
        ],
    )
