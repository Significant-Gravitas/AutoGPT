"""Unit tests for the Dakera memory blocks.

Covers the isolation/namespace logic, the missing-host and SSRF fail-fast
guards, and the store/recall run paths (including empty recall and kwarg
forwarding), which the block self-tests do not exercise.
"""

from __future__ import annotations

from typing import Any

import pytest
from dakera import RecalledMemory, RecallResponse
from pydantic import SecretStr

from backend.blocks import dakera as dakera_block
from backend.blocks.dakera import (
    DakeraBase,
    RecallDakeraMemoryBlock,
    StoreDakeraMemoryBlock,
)
from backend.data.model import HostScopedCredentials

GRAPH_ID = "graph-123"
GRAPH_EXEC_ID = "exec-456"
USER_ID = "user-789"


def _creds(host: str = "https://dakera.example.com") -> HostScopedCredentials:
    return HostScopedCredentials(
        id="11111111-1111-4111-8111-111111111111",
        provider="dakera",
        host=host,
        headers={"Authorization": SecretStr("Bearer dk-test")},
        title="Test Dakera credentials",
    )


class _RecordingClient:
    """Client double that records the kwargs it receives."""

    def __init__(self, *, store_return: dict | None = None, recall_return=None):
        self.store_kwargs: dict[str, Any] | None = None
        self.recall_kwargs: dict[str, Any] | None = None
        self._store_return = store_return or {
            "id": "mem-1",
            "content": "hello",
            "memory_type": "episodic",
            "importance": 0.5,
            "created_at": "2026-07-10T00:00:00Z",
        }
        self._recall_return = recall_return or RecallResponse(memories=[])

    def store_memory(self, **kwargs):
        self.store_kwargs = kwargs
        return self._store_return

    def recall(self, **kwargs):
        self.recall_kwargs = kwargs
        return self._recall_return


async def _collect(block, input_data, **inject) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    async for name, value in block.run(input_data, **inject):
        outputs[name] = value
    return outputs


def _wire(block, client, *, validate=None):
    """Mock the network-touching seams on a block instance."""

    async def _ok_validate(host):
        return None

    block._validate_host = validate or _ok_validate
    block._get_client = lambda credentials: client


# --------------------------------------------------------------------------- #
# _get_client
# --------------------------------------------------------------------------- #


def test_get_client_missing_host_raises():
    with pytest.raises(ValueError, match="missing a host"):
        DakeraBase._get_client(_creds(host=""))


def test_get_client_passes_timeout(monkeypatch):
    captured: dict[str, Any] = {}

    def _fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(dakera_block, "DakeraClient", _fake_client)
    DakeraBase._get_client(_creds(host="https://dakera.example.com"))
    assert captured["timeout"] == dakera_block.DEFAULT_TIMEOUT
    assert captured["base_url"] == "https://dakera.example.com"


def test_get_client_warns_on_plaintext_remote_host(monkeypatch, caplog):
    monkeypatch.setattr(dakera_block, "DakeraClient", lambda **kwargs: object())
    with caplog.at_level("WARNING"):
        DakeraBase._get_client(_creds(host="http://dakera.example.com"))
    assert any("cleartext" in r.message for r in caplog.records)


def test_get_client_no_warn_on_loopback(monkeypatch, caplog):
    monkeypatch.setattr(dakera_block, "DakeraClient", lambda **kwargs: object())
    with caplog.at_level("WARNING"):
        DakeraBase._get_client(_creds(host="http://localhost:3000"))
    assert not any("cleartext" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# _resolve_agent_id  (isolation guarantee)
# --------------------------------------------------------------------------- #


def test_resolve_agent_id_explicit_wins():
    assert DakeraBase._resolve_agent_id("team-kb", GRAPH_ID, USER_ID) == "team-kb"


def test_resolve_agent_id_strips_whitespace_then_falls_back():
    assert (
        DakeraBase._resolve_agent_id("   ", GRAPH_ID, USER_ID)
        == f"{USER_ID}:{GRAPH_ID}"
    )


def test_resolve_agent_id_empty_falls_back_to_user_and_graph():
    assert (
        DakeraBase._resolve_agent_id("", GRAPH_ID, USER_ID) == f"{USER_ID}:{GRAPH_ID}"
    )


def test_resolve_agent_id_explicit_is_stripped():
    assert DakeraBase._resolve_agent_id("  team-kb  ", GRAPH_ID, USER_ID) == "team-kb"


# --------------------------------------------------------------------------- #
# _validate_host  (SSRF guard)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_validate_host_delegates_to_egress_guard(monkeypatch):
    calls: list[str] = []

    async def _fake_validate(url):
        calls.append(url)
        return None

    monkeypatch.setattr(dakera_block, "validate_url_host", _fake_validate)
    await DakeraBase._validate_host("https://dakera.example.com")
    assert calls == ["https://dakera.example.com"]


@pytest.mark.asyncio
async def test_validate_host_propagates_blocked(monkeypatch):
    async def _blocked(url):
        raise ValueError("Access to blocked or private IP address is not allowed.")

    monkeypatch.setattr(dakera_block, "validate_url_host", _blocked)
    with pytest.raises(ValueError, match="blocked or private"):
        await DakeraBase._validate_host("http://169.254.169.254")


# --------------------------------------------------------------------------- #
# StoreDakeraMemoryBlock.run
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_store_forwards_resolved_namespace_and_session():
    block = StoreDakeraMemoryBlock()
    client = _RecordingClient()
    _wire(block, client)

    outputs = await _collect(
        block,
        block.input_schema(
            content="remember this", credentials=dakera_block.TEST_CREDENTIALS_INPUT
        ),
        credentials=_creds(),
        graph_id=GRAPH_ID,
        graph_exec_id=GRAPH_EXEC_ID,
        user_id=USER_ID,
    )

    assert client.store_kwargs["agent_id"] == f"{USER_ID}:{GRAPH_ID}"
    assert client.store_kwargs["session_id"] == GRAPH_EXEC_ID
    assert client.store_kwargs["content"] == "remember this"
    assert outputs["memory_id"] == "mem-1"
    # Output is the whitelisted, normalized record.
    assert outputs["memory"] == {
        "id": "mem-1",
        "content": "hello",
        "memory_type": "episodic",
        "importance": 0.5,
        "created_at": "2026-07-10T00:00:00Z",
    }


@pytest.mark.asyncio
async def test_store_raises_error_output_on_missing_id():
    block = StoreDakeraMemoryBlock()
    client = _RecordingClient(store_return={"content": "no id here"})
    _wire(block, client)

    outputs = await _collect(
        block,
        block.input_schema(
            content="x", credentials=dakera_block.TEST_CREDENTIALS_INPUT
        ),
        credentials=_creds(),
        graph_id=GRAPH_ID,
        graph_exec_id=GRAPH_EXEC_ID,
        user_id=USER_ID,
    )
    assert "memory_id" not in outputs
    assert "did not include a memory id" in outputs["error"]


@pytest.mark.asyncio
async def test_store_ssrf_block_routes_to_error_output():
    block = StoreDakeraMemoryBlock()
    client = _RecordingClient()

    async def _blocked(host):
        raise ValueError("Access to blocked or private IP address is not allowed.")

    _wire(block, client, validate=_blocked)

    outputs = await _collect(
        block,
        block.input_schema(
            content="x", credentials=dakera_block.TEST_CREDENTIALS_INPUT
        ),
        credentials=_creds(host="http://169.254.169.254"),
        graph_id=GRAPH_ID,
        graph_exec_id=GRAPH_EXEC_ID,
        user_id=USER_ID,
    )
    assert "blocked or private" in outputs["error"]
    # The client is never invoked once the host is rejected.
    assert client.store_kwargs is None


# --------------------------------------------------------------------------- #
# RecallDakeraMemoryBlock.run
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_recall_forwards_filters_and_maps_results():
    block = RecallDakeraMemoryBlock()
    client = _RecordingClient(
        recall_return=RecallResponse(
            memories=[
                RecalledMemory(
                    id="mem-9",
                    content="dark mode",
                    memory_type="episodic",
                    importance=0.8,
                    score=0.95,
                    created_at="2026-07-10T00:00:00Z",
                )
            ]
        )
    )
    _wire(block, client)

    outputs = await _collect(
        block,
        block.input_schema(
            query="ui prefs",
            top_k=3,
            min_importance=0.5,
            credentials=dakera_block.TEST_CREDENTIALS_INPUT,
        ),
        credentials=_creds(),
        graph_id=GRAPH_ID,
        user_id=USER_ID,
    )

    assert client.recall_kwargs["agent_id"] == f"{USER_ID}:{GRAPH_ID}"
    assert client.recall_kwargs["top_k"] == 3
    assert client.recall_kwargs["min_importance"] == 0.5
    assert client.recall_kwargs["query"] == "ui prefs"
    assert outputs["count"] == 1
    assert outputs["memories"][0] == {
        "id": "mem-9",
        "content": "dark mode",
        "memory_type": "episodic",
        "importance": 0.8,
        "created_at": "2026-07-10T00:00:00Z",
        "score": 0.95,
    }


@pytest.mark.asyncio
async def test_recall_empty_namespace_returns_zero():
    block = RecallDakeraMemoryBlock()
    client = _RecordingClient(recall_return=RecallResponse(memories=[]))
    _wire(block, client)

    outputs = await _collect(
        block,
        block.input_schema(
            query="nothing stored", credentials=dakera_block.TEST_CREDENTIALS_INPUT
        ),
        credentials=_creds(),
        graph_id=GRAPH_ID,
        user_id=USER_ID,
    )
    assert outputs["memories"] == []
    assert outputs["count"] == 0
