"""A model call that fails or is stopped still costs what the provider billed.

Run the way a user's run goes, like ``agent_run_billing_test``: the graph is
saved and started through the API's handlers and an executor pod runs it,
billing through its own database service, against a fake Anthropic endpoint
on localhost. The test reads only the user's wallet afterwards.

- A call whose reply cannot be used still reported its usage, and is billed it.
- A stop while a call is in flight bills the request's estimated input tokens,
  as the provider has the prompt and no reply reports the real count.
"""

import asyncio
import math
import os
import sys
import textwrap
from typing import Literal
from uuid import uuid4

import pytest
from aiohttp import web

from backend.api.model import CreateGraph
from backend.blocks.ai_condition import AIConditionBlock
from backend.blocks.llm import AITextGeneratorBlock
from backend.data import graph as graph_model
from backend.data.block_cost_config import TOKEN_COST
from backend.data.credit import UserCredit
from backend.data.user import get_or_create_user
from backend.integrations import credentials_store
from backend.util.prompt import estimate_token_count
from backend.util.test import SpinTestServer

from .agent_run_billing_test import (
    MODEL,
    PLATFORM_ANTHROPIC_KEY,
    _free_port,
    _kill_process_tree,
    _start_run,
)

REPORTED_INPUT_TOKENS = 1_000_000
# Long enough that its estimated input costs credits.
LONG_PROMPT = "Tell me about the sea. " * 10_000

# An executor pod as in agent_run_billing_test, which also takes a stop: a
# second line on stdin sets the run's cancel event, as a stop request does.
STOPPABLE_EXECUTOR_POD = textwrap.dedent(
    """
    import sys, threading, uuid

    from backend.data import redis_client
    from backend.data.db_manager import DatabaseManager
    from backend.data.execution import GraphExecutionEntry
    from backend.executor.cluster_lock import ClusterLock
    from backend.executor.manager import execute_graph, init_worker

    entry = GraphExecutionEntry.model_validate_json(sys.stdin.readline())
    cancel = threading.Event()

    def stop_on_request():
        if sys.stdin.readline().strip() == "stop":
            cancel.set()

    threading.Thread(target=stop_on_request, daemon=True).start()
    with DatabaseManager():
        init_worker()
        lock = ClusterLock(
            redis=redis_client.get_redis(),
            key=f"exec_lock:{entry.graph_exec_id}",
            owner_id=f"billing-e2e-{uuid.uuid4()}",
        )
        assert lock.try_acquire() == lock.owner_id
        execute_graph(entry, cancel, lock)
        lock.release()
    """
)


class FakeAnthropic:
    """Answers every call ``unclear`` (a reply that bills its usage but says
    neither true nor false) or ``hang`` (takes the request, never replies)."""

    def __init__(self, reply: Literal["unclear", "hang"]) -> None:
        self.reply = reply
        self.requests = 0
        self.request_received = asyncio.Event()
        self._closing = asyncio.Event()
        self.url = ""
        self._runner: web.AppRunner | None = None

    async def __aenter__(self) -> "FakeAnthropic":
        app = web.Application()
        app.router.add_post("/v1/messages", self._messages)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", 0).start()
        host, port = self._runner.addresses[0][:2]
        self.url = f"http://{host}:{port}"
        return self

    async def __aexit__(self, *_: object) -> None:
        self._closing.set()
        if self._runner:
            await self._runner.cleanup()

    async def _messages(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests += 1
        self.request_received.set()
        if self.reply == "hang":
            await self._closing.wait()
        return web.json_response(
            {
                "id": "msg_billing_e2e",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": "Maybe, it is hard to say."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": REPORTED_INPUT_TOKENS,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            }
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_call_with_an_unusable_reply_is_billed_its_usage(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    """The reply reported its usage before the block found it could not read
    the answer, so the user pays for that usage."""
    graph = _one_node_graph(
        AIConditionBlock().id,
        {"input_value": "42", "condition": "the input is a prime number"},
    )

    async with FakeAnthropic("unclear") as model:
        paid = await _run_and_pay(server, graph, monkeypatch, model, stop=False)

    assert model.requests == 1
    assert paid == _credits(REPORTED_INPUT_TOKENS)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_run_stopped_mid_call_is_billed_the_requests_input(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    """Stopped while the model works on the request: no reply reports usage,
    and the user pays for the prompt's estimated input tokens."""
    graph = _one_node_graph(AITextGeneratorBlock().id, {"prompt": LONG_PROMPT})

    async with FakeAnthropic("hang") as model:
        paid = await _run_and_pay(server, graph, monkeypatch, model, stop=True)

    estimate = estimate_token_count(
        [{"role": "user", "content": LONG_PROMPT}], model=MODEL.value
    )
    assert model.requests == 1
    assert paid == _credits(estimate) > 0


def _credits(input_tokens: int) -> int:
    return math.ceil(TOKEN_COST[MODEL].input * input_tokens / 1_000_000)


def _one_node_graph(block_id: str, inputs: dict) -> graph_model.Graph:
    platform_key = {
        "id": credentials_store.anthropic_credentials.id,
        "provider": "anthropic",
        "type": "api_key",
        "title": credentials_store.anthropic_credentials.title,
    }
    node = graph_model.Node(
        block_id=block_id,
        input_default={"model": MODEL.value, "credentials": platform_key, **inputs},
    )
    return graph_model.Graph(
        name="One model call", description="Calls the model once.", nodes=[node]
    )


async def _run_and_pay(
    server: SpinTestServer,
    graph: graph_model.Graph,
    monkeypatch: pytest.MonkeyPatch,
    model: FakeAnthropic,
    *,
    stop: bool,
) -> int:
    """Run ``graph`` for a new user on an executor pod, stopping it once the
    model has the request when ``stop``, and return what the wallet paid."""
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
    saved = await server.agent_server.test_create_graph(
        CreateGraph(graph=graph), user.id
    )
    balance_before = await wallet.get_credits(user.id)

    run_entry = await _start_run(saved, user.id, monkeypatch)
    await _run_on_a_stoppable_pod(run_entry, model, stop=stop)

    return balance_before - await wallet.get_credits(user.id)


async def _run_on_a_stoppable_pod(
    run_entry: str, model: FakeAnthropic, *, stop: bool
) -> None:
    pod = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        STOPPABLE_EXECUTOR_POD,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={
            **os.environ,
            "ENABLE_CREDIT": "true",
            "DATABASE_API_PORT": str(_free_port()),
            "ANTHROPIC_API_KEY": PLATFORM_ANTHROPIC_KEY,
            "ANTHROPIC_BASE_URL": model.url,
        },
    )
    assert pod.stdin is not None and pod.stdout is not None
    # Read the pod's log as it runs, or a full pipe would stall it.
    log = asyncio.ensure_future(pod.stdout.read())
    try:
        pod.stdin.write(run_entry.encode() + b"\n")
        await pod.stdin.drain()
        if stop:
            await asyncio.wait_for(model.request_received.wait(), 300)
            pod.stdin.write(b"stop\n")
            await pod.stdin.drain()
        pod.stdin.close()
        await asyncio.wait_for(pod.wait(), 600)
        output = await log
    except BaseException:
        # A run that hangs or is cancelled must not outlive the test, and
        # neither must the database service the pod started.
        if pod.returncode is None:
            _kill_process_tree(pod.pid)
            await pod.wait()
        log.cancel()
        raise
    assert pod.returncode == 0, output.decode(errors="replace")[-6000:]
