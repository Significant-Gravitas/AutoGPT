"""A model call reaches the provider once, whichever executor worker makes it.

The copilot executor runs one event loop per worker thread
(``CoPilotProcessor.on_executor_start``). The model clients used to be
process-wide, so a worker could pick up an idle keep-alive connection that
another worker's loop had opened: the request went out, reading the reply
raised "... is bound to a different event loop", and the OpenAI SDK retried
on a new connection. The provider received and billed the call twice while
the platform saw one success and recorded one.

This drives the real getters over real HTTP from several worker loops:
session titles (``_get_aux_client``), baseline rounds (``_get_main_client``)
and ``get_openai_client`` helpers, against a local OpenAI-compatible server
that journals every request. Each seed interleaves the calls randomly across
the workers, runs some concurrently, and shuts one worker down and replaces
it mid-run, the two ways a worker loop ends: stopped as
``CoPilotProcessor.cleanup`` does, or shut down as ``asyncio.run`` does.

A client that still reaches a second loop must fail there before it sends:
the last test puts the process-wide title client back and checks that.
"""

import asyncio
import concurrent.futures
import functools
import gc
import json
import logging
import random
import re
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Coroutine, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from openai.types.chat import ChatCompletion

from backend.copilot import service
from backend.copilot.baseline import service as baseline_service
from backend.copilot.config import ChatConfig
from backend.util import clients

SEEDS = range(200)
WORKERS = 3
REPLY = "Seeded reply"
# Signs that a call touched another loop's connection, or a dead loop's, or
# that a client refused to (backend/util/per_loop_cache.py same_loop_hooks).
CROSS_LOOP_SIGNS = (
    "different event loop",
    "attached to a different loop",
    "Event loop is closed",
    "Retrying request",
    "another event loop",
)

_CALL_TAG = re.compile(r"\[\[call:(\d+) delay:(\d+)\]\]")


class _ProviderHandler(BaseHTTPRequestHandler):
    # HTTP/1.1 keeps connections alive, so clients reuse them across calls.
    protocol_version = "HTTP/1.1"
    # Headers and body go out as separate writes; without TCP_NODELAY each
    # reply waits ~40 ms for a delayed ACK.
    disable_nagle_algorithm = True
    server: "_Provider"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        match = _CALL_TAG.search(json.dumps(body["messages"]))
        assert match is not None, "every test request carries a call tag"
        call_id, delay_ms = int(match[1]), int(match[2])
        with self.server.lock:
            self.server.journal.append(call_id)
        time.sleep(delay_ms / 1000)
        if body.get("stream"):
            self._send(_sse_reply(body["model"]), "text/event-stream")
        else:
            self._send(json.dumps(_completion(body["model"])), "application/json")

    def _send(self, payload: str, content_type: str) -> None:
        data = payload.encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _Provider(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _ProviderHandler)
        self.lock = threading.Lock()
        self.journal: list[int] = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


def _completion(model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-seeded",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": REPLY},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
    }


def _sse_reply(model: str) -> str:
    def chunk(delta: dict[str, Any], finish_reason: str | None) -> str:
        event = {
            "id": "chatcmpl-seeded",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(event)}\n\n"

    return (
        chunk({"role": "assistant", "content": REPLY}, None)
        + chunk({}, "stop")
        + "data: [DONE]\n\n"
    )


class _WorkerLoop:
    """An executor worker's loop, set up and torn down as ``CoPilotProcessor`` does."""

    def __init__(self, loop_errors: list[str]) -> None:
        self.loop = asyncio.new_event_loop()
        self.loop.set_exception_handler(
            lambda _, context: loop_errors.append(
                f"{context.get('message')}: {context.get('exception')!r}"
            )
        )
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.in_flight: set[concurrent.futures.Future[str]] = set()

    def submit(
        self, coro: Coroutine[Any, Any, str]
    ) -> "concurrent.futures.Future[str]":
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        self.in_flight.add(future)
        future.add_done_callback(self.in_flight.discard)
        return future

    def stop(self) -> None:
        """``CoPilotProcessor.cleanup``: stop the loop, leave it unclosed."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)

    def shut_down(self) -> None:
        """``asyncio.run``'s teardown: close async generators, then the loop."""
        asyncio.run_coroutine_threadsafe(
            self.loop.shutdown_asyncgens(), self.loop
        ).result(timeout=5)
        self.stop()
        self.loop.close()


def _tagged(text: str, call_id: int, delay_ms: int) -> str:
    return f"{text} [[call:{call_id} delay:{delay_ms}]]"


async def _session_title(call_id: int, delay_ms: int) -> str:
    title, response = await service._generate_session_title(
        _tagged("Plan a trip to Lisbon", call_id, delay_ms)
    )
    # A failed call falls back to a title built from the message.
    assert response is not None, f"title call {call_id} failed and fell back"
    return title


async def _baseline_round(call_id: int, delay_ms: int) -> str:
    state = baseline_service._BaselineStreamState(model="seeded-model")
    result = await baseline_service._baseline_llm_caller(
        [{"role": "user", "content": _tagged("Hello", call_id, delay_ms)}],
        [],
        state=state,
    )
    return result.response_text or ""


async def _helper_completion(call_id: int, delay_ms: int) -> str:
    client = clients.get_openai_client(prefer_openrouter=True)
    assert client is not None
    response = await client.chat.completions.create(
        model="seeded-model",
        messages=[
            {"role": "user", "content": _tagged("Simulate a block", call_id, delay_ms)}
        ],
    )
    return response.choices[0].message.content or ""


CALLS: tuple[Callable[[int, int], Coroutine[Any, Any, str]], ...] = (
    _session_title,
    _baseline_round,
    _helper_completion,
)


class _CrossLoopSigns(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.seen: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        text = record.getMessage()
        if record.exc_info and record.exc_info[1] is not None:
            text += f" | {record.exc_info[1]!r}"
        if any(sign in text for sign in CROSS_LOOP_SIGNS):
            self.seen.append(f"{record.name}: {text}")


@pytest.fixture
def provider() -> Iterator[_Provider]:
    server = _Provider()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def local_transport(
    provider: _Provider, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Point every chat client at the provider, the way a self-hoster's
    ``CHAT_USE_LOCAL=true`` + ``CHAT_BASE_URL`` does."""
    cfg = ChatConfig(use_local=True, api_key="local-key", base_url=provider.base_url)
    monkeypatch.setattr(service, "config", cfg)
    monkeypatch.setattr(baseline_service, "config", cfg)
    monkeypatch.setattr("backend.copilot.sdk.env.config", cfg)
    _drop_cached_clients()
    yield
    _drop_cached_clients()


def _drop_cached_clients() -> None:
    service.reset_clients()
    clients.get_openai_client.cache_clear()
    clients._get_local_openai_client.cache_clear()


@pytest.fixture
def cross_loop_signs() -> Iterator[_CrossLoopSigns]:
    handler = _CrossLoopSigns()
    root = logging.getLogger()
    noisy = [logging.getLogger(name) for name in ("openai", "httpx", "asyncio")]
    levels = [logger.level for logger in noisy]
    root.addHandler(handler)
    for logger in noisy:
        logger.setLevel(logging.DEBUG)
    yield handler
    root.removeHandler(handler)
    for logger, level in zip(noisy, levels):
        logger.setLevel(level)


def test_every_call_reaches_the_provider_once(
    provider: _Provider,
    local_transport: None,
    cross_loop_signs: _CrossLoopSigns,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_errors: list[str] = []
    unraisable: list[str] = []
    monkeypatch.setattr(
        sys,
        "unraisablehook",
        lambda hook: unraisable.append(f"{hook.err_msg}: {hook.exc_value!r}"),
    )
    workers = [_WorkerLoop(loop_errors) for _ in range(WORKERS)]
    next_call_id = 0
    try:
        for seed in SEEDS:
            try:
                next_call_id = _run_seed(
                    seed, workers, provider, loop_errors, cross_loop_signs, next_call_id
                )
            except AssertionError as failure:
                raise AssertionError(f"seed {seed}: {failure}") from failure
    finally:
        for worker in workers:
            worker.shut_down()
    gc.collect()
    assert not unraisable, unraisable
    assert not loop_errors, loop_errors
    assert not cross_loop_signs.seen, cross_loop_signs.seen


def _run_seed(
    seed: int,
    workers: list[_WorkerLoop],
    provider: _Provider,
    loop_errors: list[str],
    cross_loop_signs: _CrossLoopSigns,
    first_call_id: int,
) -> int:
    rng = random.Random(seed)
    with provider.lock:
        provider.journal.clear()
    n_calls = rng.randint(3, 6)
    replace_after = rng.randrange(n_calls)
    calls: dict[int, concurrent.futures.Future[str]] = {}
    schedule: list[str] = []
    for i, call_id in enumerate(range(first_call_id, first_call_id + n_calls)):
        worker = rng.randrange(len(workers))
        call = rng.choice(CALLS)
        delay_ms = rng.randint(0, 2)
        calls[call_id] = workers[worker].submit(call(call_id, delay_ms))
        schedule.append(f"{call.__name__}#{call_id}@w{worker}")
        if rng.random() < 0.4:
            # Let some calls finish before the next goes out, so a later call
            # can find an idle connection.
            _wait(rng.choice(list(calls.values())))
        if i == replace_after:
            victim = rng.randrange(len(workers))
            schedule.append(_replace_worker(rng, workers, victim, loop_errors))

    failures = []
    for call_id, future in calls.items():
        try:
            reply = _wait(future)
        except Exception as e:
            failures.append(f"call {call_id} raised {e!r}")
            continue
        if reply != REPLY:
            failures.append(f"call {call_id} got {reply!r}")
    with provider.lock:
        received = Counter(provider.journal)
    resent = {call_id: n for call_id, n in received.items() if n != 1}
    missing = set(calls) - set(received)
    context = (
        f"seed {seed}: schedule {schedule}\n"
        f"  calls sent more than once {resent}, never sent {sorted(missing)}\n"
        f"  {failures}\n  cross-loop signs {cross_loop_signs.seen[:3]}\n"
        f"  loop errors {loop_errors[:3]}"
    )
    assert not resent and not missing and not failures, context
    assert not cross_loop_signs.seen and not loop_errors, context
    return first_call_id + n_calls


def _replace_worker(
    rng: random.Random,
    workers: list[_WorkerLoop],
    victim: int,
    loop_errors: list[str],
) -> str:
    """Retire a worker once its own calls finish; the others keep going."""
    worker = workers[victim]
    for future in list(worker.in_flight):
        _wait(future)
    if rng.random() < 0.5:
        worker.stop()
        how = "stopped"
    else:
        held = asyncio.run_coroutine_threadsafe(_loop_clients(), worker.loop)
        worker.shut_down()
        assert all(
            client.is_closed() for client in held.result(timeout=5)
        ), f"w{victim} shut down, but its clients stayed open"
        how = "shut down"
    workers[victim] = _WorkerLoop(loop_errors)
    return f"w{victim} {how} and replaced"


async def _loop_clients() -> list[Any]:
    helper = clients.get_openai_client(prefer_openrouter=True)
    assert helper is not None
    return [service._get_main_client(), service._get_aux_client(), helper]


def _wait(future: "concurrent.futures.Future[str]") -> str:
    return future.result(timeout=30)


def test_a_process_wide_client_sends_nothing_from_a_second_loop(
    provider: _Provider,
    local_transport: None,
    cross_loop_signs: _CrossLoopSigns,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Put the bug back, one title client for the whole process: the second
    worker's title request fails before it is sent, and says why, instead of
    going out on the first worker's connection and being sent again."""
    first, second = _WorkerLoop([]), _WorkerLoop([])
    try:
        shared = _run_on(first, _aux_client())
        monkeypatch.setattr(service, "_get_aux_client", functools.cache(lambda: shared))
        titled = _run_on(first, _title_reply(0))
        refused = _run_on(second, _title_reply(1))
    finally:
        first.shut_down()
        second.shut_down()
    assert titled is not None, "the first worker's title call should succeed"
    assert refused is None, "the second worker's title call should fall back"
    with provider.lock:
        assert provider.journal == [0], "the refused title request was sent"
    assert any(
        "_get_aux_client() was first used on another event loop" in sign
        for sign in cross_loop_signs.seen
    ), cross_loop_signs.seen


async def _aux_client() -> Any:
    return service._get_aux_client()


async def _title_reply(call_id: int) -> ChatCompletion | None:
    _, response = await service._generate_session_title(
        _tagged("Plan a trip to Lisbon", call_id, 0)
    )
    return response


def _run_on(worker: _WorkerLoop, coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run_coroutine_threadsafe(coro, worker.loop).result(timeout=30)
