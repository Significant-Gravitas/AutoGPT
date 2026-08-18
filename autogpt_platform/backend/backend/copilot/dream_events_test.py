"""Tests for the ``dream.operations`` SSE event variant.

Covers payload shape, the SSE wire format (``data: ... \\n\\n``), and the
``emit_dream_operations_event`` helper so the future P6 / P9 wire-in just
has to import the helper and pass an emit closure.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from backend.copilot.dream.schemas import (
    DemotionSummary,
    DreamOperationsSnapshot,
    EntityInvalidationSummary,
    WriteSummary,
)
from backend.copilot.dream_events import (
    StreamDreamOperations,
    emit_dream_operations_event,
)
from backend.copilot.response_model import ResponseType, StreamBaseResponse


# Override heavy session-scope fixtures from ``backend/conftest.py`` —
# these tests are pure pydantic / async helper checks and don't need a
# real backend. Mirrors ``stream_heartbeat_test.py``.
@pytest_asyncio.fixture(scope="session", loop_scope="session", name="server")
async def _server_noop() -> None:
    return None


@pytest_asyncio.fixture(
    scope="session", loop_scope="session", autouse=True, name="graph_cleanup"
)
async def _graph_cleanup_noop() -> AsyncIterator[None]:
    yield


def _sample_snapshot() -> DreamOperationsSnapshot:
    """Build a non-trivial snapshot exercising every field."""
    return DreamOperationsSnapshot(
        writes=[
            WriteSummary(
                edge_uuid="edge-write-1",
                content="The user prefers async tasks.",
                scope="real:global",
                confidence=0.82,
                status="active",
                source_episode_uuids=["ep-1", "ep-2"],
            ),
        ],
        proposals=[
            WriteSummary(
                edge_uuid=None,
                content="User may be open to scheduled dream passes.",
                status="tentative",
            ),
        ],
        demotions=[
            DemotionSummary(
                edge_uuid="edge-demote-1",
                reason="stale_fact",
                new_status="superseded",
                applied=True,
            ),
        ],
        entity_invalidations=[
            EntityInvalidationSummary(
                entity_uuid="ent-1",
                reason="entity_invalidated:ent-1",
                edges_touched=["edge-demote-1"],
            ),
        ],
    )


def test_event_carries_snapshot_pass_id_and_user_id():
    """Constructed event echoes every supplied identifier and the full
    snapshot payload (no fields dropped at the pydantic boundary)."""
    snapshot = _sample_snapshot()

    event = StreamDreamOperations(
        snapshot=snapshot,
        dream_pass_id="dp-123",
        user_id="user-abc",
    )

    assert event.type is ResponseType.DREAM_OPERATIONS
    assert event.dream_pass_id == "dp-123"
    assert event.user_id == "user-abc"
    assert event.snapshot == snapshot


def test_sse_line_uses_ai_sdk_data_part_envelope():
    """``to_sse`` produces a single ``data: <json>\\n\\n`` line whose
    type matches the AI SDK v5 ``data-<name>`` convention and whose
    payload is nested under ``data`` (mirrors ``StreamStatus``)."""
    snapshot = _sample_snapshot()
    event = StreamDreamOperations(
        snapshot=snapshot, dream_pass_id="dp-1", user_id="u-1"
    )

    line = event.to_sse()

    assert line.startswith("data: ")
    assert line.endswith("\n\n")

    body = json.loads(line.removeprefix("data: ").rstrip("\n"))
    assert body["type"] == "data-dream-operations"
    assert set(body["data"].keys()) == {"snapshot", "dream_pass_id", "user_id"}
    assert body["data"]["dream_pass_id"] == "dp-1"
    assert body["data"]["user_id"] == "u-1"
    # The snapshot payload must round-trip cleanly through the snapshot
    # pydantic model (no drift in list lengths or field names).
    rehydrated = DreamOperationsSnapshot.model_validate(body["data"]["snapshot"])
    assert rehydrated == snapshot


def test_event_is_a_stream_base_response_subclass():
    """The event must be recognised as a ``StreamBaseResponse`` so the
    existing emit / reconstruction pipeline accepts it without bespoke
    type handling."""
    snapshot = _sample_snapshot()
    event = StreamDreamOperations(
        snapshot=snapshot, dream_pass_id="dp-1", user_id="u-1"
    )
    assert isinstance(event, StreamBaseResponse)


@pytest.mark.asyncio
async def test_emitter_helper_pushes_one_event_via_emit_closure():
    """The helper builds the event and pushes it through the caller's
    ``emit`` closure exactly once — no double-emit, no out-of-order
    arguments."""
    received: list[StreamBaseResponse] = []

    async def emit(event: StreamBaseResponse) -> None:
        received.append(event)

    snapshot = _sample_snapshot()
    await emit_dream_operations_event(
        emit=emit,
        snapshot=snapshot,
        dream_pass_id="dp-emit",
        user_id="u-emit",
    )

    assert len(received) == 1
    pushed = received[0]
    assert isinstance(pushed, StreamDreamOperations)
    assert pushed.dream_pass_id == "dp-emit"
    assert pushed.user_id == "u-emit"
    assert pushed.snapshot == snapshot
