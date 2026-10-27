"""An agent run is charged what its model calls cost, and its run shows that cost.

The run goes the way a user's does. The graph is saved and started through
the handlers the API uses, and an executor pod runs it: a fresh process,
configured from its environment like a deployed pod, with credits on and the
platform's Anthropic key, running the executor and the database service it
bills through. The model it reaches is a fake on localhost that keeps an
invoice of every reply. Nothing in the executor or in billing is replaced; the
one seam is the queue, where this test picks up the run and hands it to its
pod instead of the shared test server's executor, which would call the real
Anthropic API. The test reads only what a user sees afterwards: their wallet
balance and the cost on the run page. It also checks the agent's tool, which
runs on the agent's own key, starts as soon as the agent calls it.
"""

import asyncio
import os
import socket
import sys
import textwrap
from uuid import uuid4

import psutil
import pytest

from backend.api.model import CreateGraph
from backend.blocks.llm import AITextGeneratorBlock, LLMModel
from backend.blocks.orchestrator import OrchestratorBlock
from backend.data import graph as graph_model
from backend.data.block_cost_config import TOKEN_COST
from backend.data.credit import UserCredit
from backend.data.execution import GraphExecutionEntry, get_graph_execution
from backend.data.user import get_or_create_user
from backend.executor import utils as executor_utils
from backend.integrations import credentials_store
from backend.util.test import SpinTestServer

from .provider import INPUT_TOKENS_PER_REPLY, FakeAnthropic

MODEL = LLMModel("claude-sonnet-5-5")
# The platform's price for one billed reply: the model's input rate, per
# million tokens, for the tokens the fake provider reports (300 credits for
# Sonnet 5.5 today).
PRICE_PER_REPLY = int(TOKEN_COST[MODEL].input * INPUT_TOKENS_PER_REPLY / 1_000_000)
# The agent calls its tool twice, then answers: three agent replies.
TOOL_CALLS = 2
# The most a tool's model call may trail the agent's request for it. A tool
# that had to wait for the agent's hold on their shared key to expire would
# start a minute after the agent took it. On time it follows in about a second,
# so this leaves a slow runner room on both sides.
TOOL_CALL_DELAY_LIMIT_S = 30
PLATFORM_ANTHROPIC_KEY = "billing-e2e-platform-key"  # pragma: allowlist secret

# An executor pod, as a deployment runs one: the executor's worker takes the
# queued run under its cluster lock, beside the database service it bills
# through, both configured from the environment.
EXECUTOR_POD = textwrap.dedent(
    """
    import sys, threading, uuid

    from backend.data import redis_client
    from backend.data.db_manager import DatabaseManager
    from backend.data.execution import GraphExecutionEntry
    from backend.executor.cluster_lock import ClusterLock
    from backend.executor.manager import execute_graph, init_worker

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


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    ("tool_calls_fail", "billed_replies"),
    [
        # Three agent replies and two from the tool: 5 x 300 = 1,500 credits.
        (False, 5),
        # The tool's model calls fail before any work, so only the agent's
        # three replies bill: 900 credits.
        (True, 3),
    ],
    ids=["tool-calls-succeed", "tool-calls-fail"],
)
async def test_an_agent_run_costs_what_its_model_calls_cost(
    server: SpinTestServer,
    monkeypatch: pytest.MonkeyPatch,
    tool_calls_fail: bool,
    billed_replies: int,
):
    # The API offers the platform's Anthropic key, as a deployment with one does.
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
        CreateGraph(graph=_agent_with_an_ai_tool()), user.id
    )
    balance_before = await wallet.get_credits(user.id)

    async with FakeAnthropic(
        tool_calls=TOOL_CALLS, tool_calls_fail=tool_calls_fail
    ) as model:
        run_entry = await _start_run(graph, user.id, monkeypatch)
        await _run_on_an_executor_pod(run_entry, model.url)

    paid = balance_before - await wallet.get_credits(user.id)
    run = await get_graph_execution(
        user.id, GraphExecutionEntry.model_validate_json(run_entry).graph_exec_id
    )

    # The run went as scripted: the agent called its tool each time, and the
    # provider billed the replies it was meant to.
    assert model.calls("tool") >= TOOL_CALLS
    assert model.billed() == billed_replies
    # The wallet paid the platform's price for exactly those replies, and the
    # run page shows what the wallet paid.
    assert paid == billed_replies * PRICE_PER_REPLY
    assert run is not None and run.stats is not None
    assert run.stats.cost == paid
    # Each tool call reached the model as soon as the agent asked for it.
    delays = model.tool_call_delays()
    assert delays and max(delays) < TOOL_CALL_DELAY_LIMIT_S, delays


def _agent_with_an_ai_tool() -> graph_model.Graph:
    platform_key = {
        "id": credentials_store.anthropic_credentials.id,
        "provider": "anthropic",
        "type": "api_key",
        "title": credentials_store.anthropic_credentials.title,
    }
    agent = graph_model.Node(
        block_id=OrchestratorBlock().id,
        input_default={
            "prompt": "Greet the user with your tool.",
            "model": MODEL.value,
            "credentials": platform_key,
            "agent_mode_max_iterations": TOOL_CALLS + 3,
        },
    )
    tool = graph_model.Node(
        block_id=AITextGeneratorBlock().id,
        input_default={"model": MODEL.value, "credentials": platform_key},
    )
    return graph_model.Graph(
        name="Agent with an AI tool",
        description="Calls an AI block as its tool.",
        nodes=[agent, tool],
        links=[
            graph_model.Link(
                source_id=agent.id,
                sink_id=tool.id,
                source_name=f"tools_^_{tool.id}_~_prompt",
                sink_name="prompt",
            )
        ],
    )


async def _start_run(
    graph: graph_model.GraphModel, user_id: str, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Start the run and return the queued message the executor would receive.

    The shared test server's executor reaches the real Anthropic API, so the
    run is taken off the queue here and handed to this test's pod instead."""
    queued: list[str] = []

    class _Queue:
        async def publish_message(self, *, message: str, **_: object) -> None:
            queued.append(message)

    async def _execution_queue() -> _Queue:
        return _Queue()

    monkeypatch.setattr(executor_utils, "get_async_execution_queue", _execution_queue)
    await executor_utils.add_graph_execution(
        graph_id=graph.id, user_id=user_id, graph_version=graph.version
    )
    (message,) = queued
    return message


async def _run_on_an_executor_pod(run_entry: str, model_url: str) -> None:
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
            "ANTHROPIC_API_KEY": PLATFORM_ANTHROPIC_KEY,
            "ANTHROPIC_BASE_URL": model_url,
        },
    )
    try:
        output, _ = await asyncio.wait_for(pod.communicate(run_entry.encode()), 600)
    except BaseException:
        # A run that hangs or is cancelled must not outlive the test, and
        # neither must the database service the pod started, a grandchild
        # through multiprocessing's forkserver.
        if pod.returncode is None:
            _kill_process_tree(pod.pid)
            await pod.wait()
        raise
    assert pod.returncode == 0, output.decode(errors="replace")[-6000:]


def _kill_process_tree(pid: int) -> None:
    try:
        root = psutil.Process(pid)
        # Collected before any kill: a killed parent's children are reparented
        # and no longer found under it.
        processes = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return
    for process in processes:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
