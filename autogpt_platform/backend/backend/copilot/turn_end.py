"""Tools that end the turn: once one succeeds, nothing runs after it.

A question for the user (``ask_question``) is the last thing in its turn: no
more text and no more tool calls, so the question card is what the user is
left looking at. Starting a background task (``start_task``) ends the turn
too, so the chat is free while the work runs, but the other calls of that
step still run (the user may have asked for several tasks at once). The tool
registers the session here when it succeeds, and each engine then:

* makes no further model call (the SDK through its PostToolUse hook, the
  baseline by leaving its tool loop), and
* after a question, refuses every tool call that starts later (calls the
  model placed before the question in the same step have already run).

Process-local: the tool runs in the same executor process as the engine that
checks. Each turn clears its session when it starts, and the executor clears
it again when the turn is over, so an entry never reaches a later turn.
"""

import threading

ENDED_REFUSAL = (
    "Not run: you asked the user a question, which ends your turn. "
    "Call this again after they answer."
)

_lock = threading.Lock()
# Session -> whether calls that start from now on are refused.
_ending: dict[str, bool] = {}


def request_end(session_id: str, *, refuse_later_calls: bool = True) -> None:
    """End *session_id*'s turn after the current tool step; with
    *refuse_later_calls*, also refuse every call that starts from now on."""
    with _lock:
        _ending[session_id] = _ending.get(session_id, False) or refuse_later_calls


def is_ending(session_id: str | None) -> bool:
    """True when a tool asked to end *session_id*'s current turn."""
    if not session_id:
        return False
    with _lock:
        return session_id in _ending


def refuses_calls(session_id: str | None) -> bool:
    """True when calls that start now are refused: the turn asked a question."""
    if not session_id:
        return False
    with _lock:
        return _ending.get(session_id, False)


def clear(session_id: str) -> None:
    """Forget the request, at the start and end of every turn."""
    with _lock:
        _ending.pop(session_id, None)
