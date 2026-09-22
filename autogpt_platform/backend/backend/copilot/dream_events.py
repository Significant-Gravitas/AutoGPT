"""SSE events emitted when a dream pass produces operations.

Lives in its own module so ``response_model.py`` stays under the 300-line
target. The event mirrors the ``data-status`` / ``data-cursor`` shape
already in use by the AI SDK v5 wire protocol — a ``data-<name>`` part
with an opaque ``data`` payload — so the frontend just needs to recognise
a new ``part.type`` rather than learn a whole new envelope.

Per ``dream/p0-spec.md`` §11 item #11 and ``dream/TODO.md`` P9 note: the
emitter helper is provided here but NOT wired into the dream orchestrator
yet. P6 (surface dreams in chat UI) and P9 (daydreaming) own the wire-in
so the gating (``mode in {dream, daydream}``) lives next to the
orchestrator state machine, not duplicated here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import Field

from backend.copilot.dream.schemas import DreamOperationsSnapshot
from backend.copilot.response_model import ResponseType, StreamBaseResponse

logger = logging.getLogger(__name__)


class StreamDreamOperations(StreamBaseResponse):
    """Structured snapshot of a completed dream pass, surfaced inline on
    the chat stream so P6 (surface in chat) and P9 (daydreaming) can
    render a "dream summary" artifact without a separate admin fetch.

    The frontend Vercel AI SDK accumulates this as a
    ``type="data-dream-operations"`` UIMessage part with the snapshot
    available under ``part.data.snapshot``. Bookkeeping fields
    (``dream_pass_id``, ``user_id``) live alongside so a consumer can
    reconcile against the admin endpoint or AgentProbe scoring without
    re-querying.
    """

    type: ResponseType = Field(
        default=ResponseType.DREAM_OPERATIONS,
        description="AI SDK v5 data part wire type.",
    )
    snapshot: DreamOperationsSnapshot = Field(
        ..., description="Per-operation rollup produced by phase 3 + apply.py."
    )
    dream_pass_id: str = Field(
        ..., description="Identifier for the dream pass that produced this snapshot."
    )
    user_id: str = Field(..., description="Owner of the dream pass.")

    def to_sse(self) -> str:
        """Emit as an AI SDK v5 ``data-<name>`` part.

        Mirrors ``StreamStatus.to_sse`` — payload nests under ``data`` so
        the SDK surfaces it on ``message.parts`` as a recognised data
        part instead of dropping it as unknown.
        """
        payload: dict[str, Any] = {
            "type": self.type.value,
            "data": {
                "snapshot": self.snapshot.model_dump(mode="json"),
                "dream_pass_id": self.dream_pass_id,
                "user_id": self.user_id,
            },
        }
        return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Emitter helper
# ---------------------------------------------------------------------------


EmitFn = Callable[[StreamBaseResponse], Awaitable[None]]


async def emit_dream_operations_event(
    emit: EmitFn,
    snapshot: DreamOperationsSnapshot,
    dream_pass_id: str,
    user_id: str,
) -> None:
    """Push a :class:`StreamDreamOperations` event onto the active stream.

    The orchestrator / daydream path passes its own ``emit`` closure (the
    same one baseline / sdk use for ``StreamStatus``) so the event lands
    on the user's open SSE subscription and gets persisted via the Redis
    snapshot pipeline alongside every other turn event.
    """
    event = StreamDreamOperations(
        snapshot=snapshot,
        dream_pass_id=dream_pass_id,
        user_id=user_id,
    )
    await emit(event)
