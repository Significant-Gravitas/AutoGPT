"""Tests for the Sentry tracing convention in ``backend.util.tracing``.

Most run against a live SDK client (the ``sentry_transactions`` fixture) and
assert on the transaction payloads it would have sent, so they pin what lands
in Sentry rather than which SDK functions were called.
"""

import asyncio
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
import sentry_sdk
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.tracing import NoOpSpan

from backend.util import tracing

# ---------- traces_sampler ----------


def _http(path: str, *, parent_sampled: bool | None = None) -> dict:
    return {
        "transaction_context": {"op": "http.server", "name": path},
        "parent_sampled": parent_sampled,
        "asgi_scope": {"type": "http", "path": path},
    }


@pytest.mark.parametrize(
    "path",
    [
        "/metrics",
        "/",
        "/health",
        "/health_check",
        "/health_check_async",
        "/api/chat/health",
        "/healthz",
    ],
)
def test_probes_and_scrapes_are_not_traced(path: str) -> None:
    assert tracing.traces_sampler(_http(path)) == 0.0


def test_a_websocket_connection_is_not_traced() -> None:
    context = {
        "transaction_context": {"op": "websocket.server", "name": "/ws"},
        "parent_sampled": None,
        "asgi_scope": {"type": "websocket", "path": "/ws"},
    }

    assert tracing.traces_sampler(context) == 0.0


def test_a_websocket_is_recognised_by_op_without_an_asgi_scope() -> None:
    context = {"transaction_context": {"op": "websocket.server", "name": "/ws"}}

    assert tracing.traces_sampler(context) == 0.0


@pytest.mark.parametrize(
    "name",
    ["http://10.0.0.5:8006/metrics", "http://10.0.0.5:8006/health?probe=1"],
)
def test_a_probe_named_after_its_full_url_is_not_traced(name: str) -> None:
    """Before routing, an HTTP transaction is named after its raw URL."""
    context = {
        "transaction_context": {"op": "http.server", "name": name},
        "parent_sampled": None,
    }

    assert tracing.traces_sampler(context) == 0.0


def test_a_probe_on_an_aiohttp_server_is_not_traced() -> None:
    """aiohttp names a transaction generically until its route resolves; the
    request it hands the sampler has the path."""
    context = {
        "transaction_context": {
            "op": "http.server",
            "name": "generic AIOHTTP request",
        },
        "parent_sampled": None,
        "aiohttp_request": SimpleNamespace(path="/healthz"),
    }

    assert tracing.traces_sampler(context) == 0.0
    context["aiohttp_request"] = SimpleNamespace(path="/v1/messages")
    assert tracing.traces_sampler(context) == 1.0


def test_a_dropped_endpoint_stays_dropped_even_with_a_sampled_parent() -> None:
    assert tracing.traces_sampler(_http("/metrics", parent_sampled=True)) == 0.0


@pytest.mark.parametrize(
    "path",
    [
        "/api/store/agents/alice/health",
        "/api/store/agents/alice/readyz",
        "/api/store/agents/alice/healthz",
        "/api/store/agents/alice/livez",
        "/api/store/agents/alice/readiness",
        "/api/store/agents/alice/health_check",
        "/api/store/agents/alice/metrics",
        "/health/",
    ],
)
@pytest.mark.parametrize("parent_sampled", [None, True])
def test_a_route_that_only_ends_like_a_probe_is_traced(
    path: str, parent_sampled: bool | None
) -> None:
    """Only the exact probe paths are dropped: a store agent slugged "health"
    is real traffic."""
    context = _http(path, parent_sampled=parent_sampled)
    url_named = {
        "transaction_context": {
            "op": "http.server",
            "name": f"http://10.0.0.5:8006{path}",
        },
        "parent_sampled": parent_sampled,
    }

    assert tracing.traces_sampler(context) == 1.0
    assert tracing.traces_sampler(url_named) == 1.0


def test_through_the_asgi_middleware_only_the_probe_is_dropped(
    sentry_transactions,
) -> None:
    """End to end through the SDK's ASGI middleware, as each service runs."""
    app = FastAPI()

    @app.get("/health")
    def health() -> dict:
        return {"status": "healthy"}

    @app.get("/api/store/agents/{username}/{agent_name}")
    def agent(username: str, agent_name: str) -> dict:
        return {"agent": agent_name}

    client = TestClient(SentryAsgiMiddleware(app, transaction_style="url"))
    for path in (
        "/health",
        "/api/store/agents/alice/health",
        "/api/store/agents/alice/readyz",
    ):
        assert client.get(path).status_code == 200

    assert [t["request"]["url"] for t in sentry_transactions] == [
        "http://testserver/api/store/agents/alice/health",
        "http://testserver/api/store/agents/alice/readyz",
    ]
    assert {t["transaction"] for t in sentry_transactions} == {
        "/api/store/agents/{username}/{agent_name}"
    }


@pytest.mark.parametrize("parent_sampled, expected", [(True, 1.0), (False, 0.0)])
def test_the_parents_decision_is_kept(parent_sampled: bool, expected: float) -> None:
    context = _http("/api/graphs", parent_sampled=parent_sampled)

    assert tracing.traces_sampler(context) == expected


@pytest.mark.parametrize(
    "context",
    [
        _http("/api/chat/sessions/{session_id}/stream"),
        _http("/update_node_execution_status"),
        {
            "transaction_context": {
                "op": "executor.node",
                "name": "AITextGeneratorBlock",
            }
        },
        {"transaction_context": {"op": "copilot.turn", "name": "copilot.turn"}},
    ],
)
def test_everything_else_is_traced(context: dict) -> None:
    assert tracing.traces_sampler(context) == 1.0


def test_a_malformed_sampling_context_is_traced_rather_than_raising() -> None:
    assert tracing.traces_sampler({"transaction_context": "not-a-dict"}) == 1.0


# ---------- start_transaction ----------


def _trace(transaction: dict) -> dict:
    return transaction["contexts"]["trace"]


def test_a_unit_of_work_becomes_one_transaction(sentry_transactions) -> None:
    with tracing.start_transaction(
        "copilot.turn",
        "copilot.turn",
        tags={"copilot.engine": "sdk"},
        data={"copilot.session_id": "s-1"},
    ):
        with sentry_sdk.start_span(op="db.redis", name="GET"):
            pass

    [transaction] = sentry_transactions
    assert transaction["transaction"] == "copilot.turn"
    assert transaction["transaction_info"] == {"source": "task"}
    assert _trace(transaction)["op"] == "copilot.turn"
    assert _trace(transaction)["status"] == "ok"
    assert _trace(transaction)["data"]["copilot.session_id"] == "s-1"
    assert transaction["tags"]["copilot.engine"] == "sdk"
    assert [span["op"] for span in transaction["spans"]] == ["db.redis"]


def test_handed_over_trace_headers_are_continued(sentry_transactions) -> None:
    trace_id, parent_span_id = "a" * 32, "b" * 16

    with tracing.start_transaction(
        "copilot.turn",
        "copilot.turn",
        trace_headers={"sentry-trace": f"{trace_id}-{parent_span_id}-1"},
    ):
        pass

    [transaction] = sentry_transactions
    assert _trace(transaction)["trace_id"] == trace_id
    assert _trace(transaction)["parent_span_id"] == parent_span_id


def test_an_unsampled_parent_is_not_traced(sentry_transactions) -> None:
    with tracing.start_transaction(
        "copilot.turn",
        "copilot.turn",
        trace_headers={"sentry-trace": f"{'a' * 32}-{'b' * 16}-0"},
    ):
        pass

    assert sentry_transactions == []


def test_a_keyed_unit_joins_the_trace_for_its_key(sentry_transactions) -> None:
    trace_id = tracing.trace_id_for(str(uuid.uuid4()))

    for _ in range(2):
        with tracing.start_transaction("executor.node", "Block", trace_id=trace_id):
            pass

    assert [_trace(t)["trace_id"] for t in sentry_transactions] == [trace_id] * 2
    assert all(_trace(t).get("parent_span_id") is None for t in sentry_transactions)


def test_a_unit_inside_another_becomes_its_child(sentry_transactions) -> None:
    with tracing.start_transaction("copilot.turn", "copilot.turn") as turn:
        with tracing.start_transaction("copilot.tool", "run_block"):
            with sentry_sdk.start_span(op="http.client", name="POST /rpc"):
                pass
        with sentry_sdk.start_span(op="db.redis", name="XADD"):
            pass

    tool, outer = sentry_transactions
    assert tool["transaction"] == "run_block"
    assert _trace(tool)["trace_id"] == turn.trace_id
    assert _trace(tool)["parent_span_id"] == turn.span_id
    assert [span["op"] for span in tool["spans"]] == ["http.client"]
    assert [span["op"] for span in outer["spans"]] == ["db.redis"]


def test_handed_over_headers_win_over_the_running_transaction(
    sentry_transactions,
) -> None:
    trace_id = "c" * 32

    with tracing.start_transaction("copilot.turn", "copilot.turn"):
        with tracing.start_transaction(
            "copilot.turn",
            "copilot.turn",
            trace_headers={"sentry-trace": f"{trace_id}-{'d' * 16}-1"},
        ):
            pass

    inner, _ = sentry_transactions
    assert _trace(inner)["trace_id"] == trace_id


def test_the_caller_scope_is_left_as_it_was(sentry_transactions) -> None:
    scope = sentry_sdk.get_current_scope()
    before = scope.span

    with tracing.start_transaction("executor.node", "Block"):
        sentry_sdk.set_tag("executor.leaked", "yes")

    assert sentry_sdk.get_current_scope().span is before
    with tracing.start_transaction("executor.node", "Block"):
        pass
    assert "executor.leaked" not in sentry_transactions[-1].get("tags", {})


async def test_concurrent_units_sharing_a_scope_keep_their_own_spans(
    sentry_transactions,
) -> None:
    """Coroutines handed to another loop with run_coroutine_threadsafe share
    one Sentry scope; each unit's spans must still land on its own transaction."""
    both_started = asyncio.Event()
    started = 0

    async def unit(name: str) -> None:
        nonlocal started
        with tracing.start_transaction("executor.node", name):
            started += 1
            if started == 2:
                both_started.set()
            await both_started.wait()
            with sentry_sdk.start_span(op="http.client", name=f"call from {name}"):
                await asyncio.sleep(0)

    await asyncio.gather(unit("A"), unit("B"))

    by_name = {t["transaction"]: t for t in sentry_transactions}
    assert [s["description"] for s in by_name["A"]["spans"]] == ["call from A"]
    assert [s["description"] for s in by_name["B"]["spans"]] == ["call from B"]


def test_an_exception_fails_the_transaction_and_propagates(
    sentry_transactions,
) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with tracing.start_transaction("executor.node", "Block"):
            raise RuntimeError("boom")

    [transaction] = sentry_transactions
    assert _trace(transaction)["status"] == "internal_error"


async def test_a_cancellation_is_recorded_as_cancelled(sentry_transactions) -> None:
    async def turn() -> None:
        with tracing.start_transaction("copilot.turn", "copilot.turn"):
            await asyncio.sleep(10)

    task = asyncio.create_task(turn())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    [transaction] = sentry_transactions
    assert _trace(transaction)["status"] == "cancelled"


def test_a_status_set_by_the_block_is_kept(sentry_transactions) -> None:
    with tracing.start_transaction("executor.node", "Block"):
        with sentry_sdk.start_span(op="function", name="inner"):
            tracing.set_status("invalid_argument")

    assert _trace(sentry_transactions[0])["status"] == "invalid_argument"


def test_without_a_sentry_client_the_block_runs_untraced() -> None:
    assert not sentry_sdk.get_client().is_active()
    ran = False

    with tracing.start_transaction("executor.node", "Block") as span:
        ran = True

    assert ran
    assert isinstance(span, NoOpSpan)


def test_a_transaction_that_cannot_start_leaves_the_block_running(
    sentry_transactions, monkeypatch
) -> None:
    def refuse(*args, **kwargs):
        raise RuntimeError("sdk failure")

    monkeypatch.setattr(sentry_sdk, "continue_trace", refuse)
    ran = False

    with tracing.start_transaction("executor.node", "Block") as span:
        ran = True

    assert ran
    assert isinstance(span, NoOpSpan)
    assert sentry_transactions == []


# ---------- timings and data ----------


def test_timings_land_on_the_current_transaction(sentry_transactions) -> None:
    with tracing.start_transaction("copilot.turn", "copilot.turn"):
        with sentry_sdk.start_span(op="function", name="inner"):
            tracing.set_timing("copilot.turn.ttft_ms", 1234.56789)
            tracing.set_data("copilot.turn.chunk_count", 7)

    data = _trace(sentry_transactions[0])["data"]
    assert data["copilot.turn.ttft_ms"] == 1234.568
    assert data["copilot.turn.chunk_count"] == 7


def test_a_negative_duration_is_recorded_as_zero(sentry_transactions) -> None:
    with tracing.start_transaction("executor.node", "Block") as span:
        tracing.set_timing("executor.node.queue_wait_ms", -3.0, span=span)

    assert _trace(sentry_transactions[0])["data"]["executor.node.queue_wait_ms"] == 0


def test_a_value_that_is_not_a_duration_is_ignored(sentry_transactions) -> None:
    not_a_duration: Any = "soon"

    with tracing.start_transaction("copilot.turn", "copilot.turn"):
        tracing.set_timing("copilot.turn.ttft_ms", not_a_duration)

    assert "copilot.turn.ttft_ms" not in _trace(sentry_transactions[0])["data"]


def test_a_timing_outside_a_transaction_is_ignored(sentry_transactions) -> None:
    tracing.set_timing("copilot.turn.ttft_ms", 10.0)

    assert sentry_transactions == []


# ---------- suppress_child_spans ----------


def test_suppressed_spans_are_not_recorded(sentry_transactions) -> None:
    with tracing.start_transaction("copilot.turn", "copilot.turn"):
        with tracing.suppress_child_spans():
            for _ in range(5):
                with sentry_sdk.start_span(op="db.redis", name="XREAD"):
                    pass
        with sentry_sdk.start_span(op="db.redis", name="HGETALL"):
            pass

    [transaction] = sentry_transactions
    assert [span["description"] for span in transaction["spans"]] == ["HGETALL"]


def test_suppression_keeps_errors_on_the_transactions_trace(
    sentry_transactions,
) -> None:
    with tracing.start_transaction("copilot.turn", "copilot.turn") as span:
        with tracing.suppress_child_spans():
            trace_context = sentry_sdk.get_current_scope().get_trace_context()

    assert trace_context["trace_id"] == span.trace_id


def test_suppression_without_a_transaction_is_harmless(sentry_transactions) -> None:
    with tracing.suppress_child_spans():
        with sentry_sdk.start_span(op="db.redis", name="GET"):
            pass

    assert sentry_transactions == []


# ---------- trace_propagation_headers ----------


def test_no_headers_are_handed_over_outside_a_transaction(
    sentry_transactions,
) -> None:
    assert tracing.trace_propagation_headers() == {}


def test_headers_carry_the_active_trace(sentry_transactions) -> None:
    with tracing.start_transaction("copilot.turn", "copilot.turn") as span:
        headers = tracing.trace_propagation_headers()

    assert headers["sentry-trace"] == f"{span.trace_id}-{span.span_id}-1"
    assert f"sentry-trace_id={span.trace_id}" in headers["baggage"]


def test_no_headers_without_a_sentry_client() -> None:
    assert tracing.trace_propagation_headers() == {}


# ---------- trace_id_for ----------


def test_a_uuid_key_is_its_own_trace_id() -> None:
    key = uuid.uuid4()

    assert tracing.trace_id_for(str(key)) == key.hex


def test_any_other_key_maps_to_a_stable_trace_id() -> None:
    trace_id = tracing.trace_id_for("graph-exec-1")

    assert trace_id == tracing.trace_id_for("graph-exec-1")
    assert trace_id != tracing.trace_id_for("graph-exec-2")
    assert len(trace_id) == 32
    int(trace_id, 16)
