"""A block run from an AutoPilot chat costs what it costs in a workflow.

The AI text block on an OpenRouter model is priced on what each call costs
the platform: OpenRouter reports the cost and the platform charges 150
credits per dollar of it. The block runs three times as a workflow's nodes,
on an executor pod as in ``agent_run_billing_test``, and three times from a
chat's ``run_block`` tool, each time on the platform's OpenRouter key. The
model is a fake on localhost that bills every call $0.075, which is 11.25
credits, charged as 12. Nothing in the executor, the tool or billing is
replaced; the test reads only the wallets, the run's cost and the fake's own
count of the calls it served.

An empty wallet can't start such a block from a chat either.
"""

import asyncio
import os
import sys
import textwrap
from typing import Any, Self
from uuid import uuid4

import pytest
from aiohttp import web
from pydantic import SecretStr

from backend.api.model import CreateGraph
from backend.blocks.ai_condition import AIConditionBlock
from backend.blocks.llm import AITextGeneratorBlock, LLMModel
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.models import (
    BlockOutputResponse,
    ErrorResponse,
    ToolResponseBase,
)
from backend.copilot.tools.run_block import RunBlockTool
from backend.data import credit
from backend.data import graph as graph_model
from backend.data.credit import UserCredit
from backend.data.execution import GraphExecutionEntry, get_graph_execution
from backend.data.user import get_or_create_user
from backend.integrations import credentials_store
from backend.util.llm import providers
from backend.util.test import SpinTestServer

from .agent_run_billing_test import _free_port, _kill_process_tree, _start_run

MODEL = LLMModel("cohere/command-a-03-2025")
COST_PER_CALL_USD = 0.075
CREDITS_PER_RUN = 12
RUNS = 3
PLATFORM_OPENROUTER_KEY = "billing-e2e-openrouter-key"  # pragma: allowlist secret

# An executor pod, as in agent_run_billing_test, whose OpenRouter is the
# fake: the client's address is a constant, so the pod points it there.
EXECUTOR_POD = textwrap.dedent(
    """
    import os, sys, threading, uuid

    from backend.data import redis_client
    from backend.data.db_manager import DatabaseManager
    from backend.data.execution import GraphExecutionEntry
    from backend.executor.cluster_lock import ClusterLock
    from backend.executor.manager import execute_graph, init_worker
    from backend.util.llm import providers

    providers.OPENROUTER_BASE_URL = os.environ["BILLING_E2E_OPENROUTER_URL"]
    entry = GraphExecutionEntry.model_validate_json(sys.stdin.read())
    with DatabaseManager():
        init_worker()
        lock = ClusterLock(
            redis=redis_client.get_redis(),
            key=f"exec_lock:{entry.graph_exec_id}",
            owner_id=f"billing-e2e-{uuid.uuid4()}",
        )
        assert lock.try_acquire() == lock.owner_id
        execute_graph(entry, threading.Event(), lock)
        lock.release()
    """
)


class FakeOpenRouter:
    """OpenRouter's chat completions on localhost. Every call is answered and
    reports its cost on ``usage.cost``, as OpenRouter does when asked to."""

    def __init__(self) -> None:
        self.calls = 0
        self.url = ""
        self._runner: web.AppRunner | None = None

    async def __aenter__(self) -> Self:
        app = web.Application()
        app.router.add_post("/api/v1/chat/completions", self._chat)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", 0).start()
        host, port = self._runner.addresses[0][:2]
        self.url = f"http://{host}:{port}/api/v1"
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _chat(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.calls += 1
        return web.json_response(
            {
                "id": f"gen-{uuid4().hex}",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1_000,
                    "completion_tokens": 10,
                    "total_tokens": 1_010,
                    "cost": COST_PER_CALL_USD,
                },
            }
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_block_priced_on_usage_costs_the_same_from_a_chat_as_in_a_workflow(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    _bill_credits_on_the_platform_key(monkeypatch)
    wallet = UserCredit()
    workflow_user, chat_user = await _new_user(), await _new_user()
    for user_id in (workflow_user, chat_user):
        await wallet.grant_credits(user_id, 1_000, "billing e2e")
    graph = await server.agent_server.test_create_graph(
        CreateGraph(graph=_text_nodes(RUNS)), workflow_user
    )

    async with FakeOpenRouter() as openrouter:
        monkeypatch.setattr(providers, "OPENROUTER_BASE_URL", openrouter.url)
        run_entry = await _start_run(graph, workflow_user, monkeypatch)
        await _run_on_an_executor_pod(run_entry, openrouter.url)
        chat_replies = [
            await _run_from_a_chat(
                chat_user, AITextGeneratorBlock(), {"prompt": f"Say hi to {n}."}
            )
            for n in range(RUNS)
        ]

    run = await get_graph_execution(
        workflow_user, GraphExecutionEntry.model_validate_json(run_entry).graph_exec_id
    )
    # Every run reached the model: three nodes and three chat calls.
    assert openrouter.calls == 2 * RUNS
    assert all(isinstance(reply, BlockOutputResponse) for reply in chat_replies)
    # 3 runs at 12 credits each: both wallets paid 36, and the run page
    # shows what the workflow's wallet paid.
    assert await wallet.get_credits(workflow_user) == 1_000 - RUNS * CREDITS_PER_RUN
    assert await wallet.get_credits(chat_user) == 1_000 - RUNS * CREDITS_PER_RUN
    assert run is not None and run.stats is not None
    assert run.stats.cost == RUNS * CREDITS_PER_RUN


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    ("block_cls", "input_data"),
    [
        (AITextGeneratorBlock, {"prompt": "Say hi."}),
        # A block with no estimate of its cost before it runs.
        (AIConditionBlock, {"input_value": "7", "condition": "is prime"}),
    ],
)
async def test_an_empty_wallet_cannot_run_a_block_priced_on_usage_from_a_chat(
    server: SpinTestServer,
    monkeypatch: pytest.MonkeyPatch,
    block_cls: type,
    input_data: dict[str, Any],
):
    _bill_credits_on_the_platform_key(monkeypatch)
    wallet = UserCredit()
    user_id = await _new_user()
    assert await wallet.get_credits(user_id) == 0

    async with FakeOpenRouter() as openrouter:
        monkeypatch.setattr(providers, "OPENROUTER_BASE_URL", openrouter.url)
        reply = await _run_from_a_chat(user_id, block_cls(), input_data)

    assert isinstance(reply, ErrorResponse)
    assert "Insufficient credits" in reply.message
    assert openrouter.calls == 0
    assert await wallet.get_credits(user_id) == 0


def _bill_credits_on_the_platform_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Credits on, and the platform's OpenRouter key offered to every user,
    as a deployment with one does."""
    monkeypatch.setattr(credit.settings.config, "enable_credit", True)
    monkeypatch.setattr(
        credentials_store.settings.secrets,
        "open_router_api_key",
        PLATFORM_OPENROUTER_KEY,
    )
    monkeypatch.setattr(
        credentials_store.open_router_credentials,
        "api_key",
        SecretStr(PLATFORM_OPENROUTER_KEY),
    )


async def _new_user() -> str:
    user = await get_or_create_user(
        {
            "sub": str(uuid4()),
            "email": f"billing-e2e-{uuid4()}@example.com",
            "name": "Billing E2E",
        }
    )
    return user.id


def _text_nodes(count: int) -> graph_model.Graph:
    platform_key = {
        "id": credentials_store.open_router_credentials.id,
        "provider": "open_router",
        "type": "api_key",
        "title": credentials_store.open_router_credentials.title,
    }
    return graph_model.Graph(
        name="Model calls",
        description="Calls the model once from each node.",
        nodes=[
            graph_model.Node(
                block_id=AITextGeneratorBlock().id,
                input_default={
                    "model": MODEL.value,
                    "prompt": f"Say hi to {n}.",
                    "credentials": platform_key,
                },
            )
            for n in range(count)
        ],
    )


async def _run_from_a_chat(
    user_id: str, block: Any, input_data: dict[str, Any]
) -> ToolResponseBase:
    """Run ``block`` as a chat's run_block call does, with the credential
    the tool picks for the user: the platform's."""
    return await RunBlockTool()._execute(
        user_id=user_id,
        session=make_session(user_id),
        block_id=block.id,
        input_data={**input_data, "model": MODEL.value},
    )


async def _run_on_an_executor_pod(run_entry: str, openrouter_url: str) -> None:
    pod = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        EXECUTOR_POD,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={
            **os.environ,
            "ENABLE_CREDIT": "true",
            "DATABASE_API_PORT": str(_free_port()),
            "OPEN_ROUTER_API_KEY": PLATFORM_OPENROUTER_KEY,
            "BILLING_E2E_OPENROUTER_URL": openrouter_url,
        },
    )
    try:
        output, _ = await asyncio.wait_for(pod.communicate(run_entry.encode()), 600)
    except BaseException:
        # A run that hangs or is cancelled must not outlive the test, and
        # neither must the database service the pod started.
        if pod.returncode is None:
            _kill_process_tree(pod.pid)
            await pod.wait()
        raise
    assert pod.returncode == 0, output.decode(errors="replace")[-6000:]
