"""In-process registry for baseline→SDK engine switches.

``enter_agent_building_mode``, when called on a baseline turn of an
SDK-capable deployment, registers the session here. The baseline service
ends its tool loop at the next iteration boundary; the executor MANAGER —
in the turn future's done-callback, after the turn thread has fully
finished (fail-close ran, RMQ message acked, cluster lock released,
active_tasks cleaned) — pops the entry and dispatches a continuation turn
on the SDK engine. Dispatching any earlier trips the turn-scoped
fail-close CAS and the RMQ same-session duplicate rejection.

Process-local by design: registration (baseline service) and consumption
(manager done-callback) both run inside the same executor process, so the
dict never needs to cross process boundaries. If the backend restarts
between registration and dispatch, the continuation is lost, but the
session still switches engines on the next user message via the
processor's derived building-mode check — graceful degradation, no
persistence needed. Entries cannot go stale: every registered turn ends in
its own done-callback, which pops unconditionally.
"""

import threading

from pydantic import BaseModel, ConfigDict

# Dispatched as the continuation turn's message with is_user_message=False,
# so it persists as an assistant row and renders as AutoPilot narration —
# phrase it in assistant voice (it doubles as the model's continuation
# prompt; the guide itself arrives via the system prompt). Sibling of the
# SDK in-turn restart prompt (_BUILDING_MODE_CONTINUATION in sdk/service.py)
# — keep the two aligned when rewording.
CONTINUATION_MESSAGE = (
    "Building mode is active — the agent-building guide is loaded. "
    "Continuing with the request."
)


class SwitchRequest(BaseModel):
    """Tenancy context needed to dispatch the continuation turn."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    organization_id: str | None = None
    team_id: str | None = None


_lock = threading.Lock()
_pending: dict[str, SwitchRequest] = {}


def request_switch(
    session_id: str,
    *,
    user_id: str,
    organization_id: str | None = None,
    team_id: str | None = None,
) -> None:
    """Register *session_id* for an engine-switch continuation."""
    with _lock:
        _pending[session_id] = SwitchRequest(
            user_id=user_id,
            organization_id=organization_id,
            team_id=team_id,
        )


def is_pending(session_id: str) -> bool:
    """True when a switch is registered for *session_id* (non-consuming)."""
    with _lock:
        return session_id in _pending


def pop_switch(session_id: str) -> SwitchRequest | None:
    """Consume the pending switch for *session_id*, or None."""
    with _lock:
        return _pending.pop(session_id, None)
