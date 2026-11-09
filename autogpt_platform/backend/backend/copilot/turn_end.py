"""Tools that end the turn: once one succeeds, nothing runs after it.

A question for the user (``ask_question``) is the last thing in its turn: no
more text and no more tool calls, so the question card is what the user is
left looking at. The tool registers the session here when it succeeds, and
each engine then:

* refuses every tool call that starts after it (calls the model placed
  before the question in the same step have already run), and
* makes no further model call (the SDK through its PostToolUse hook, the
  baseline by leaving its tool loop).

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
_ending: set[str] = set()


def request_end(session_id: str) -> None:
    """End *session_id*'s turn after the current tool step."""
    with _lock:
        _ending.add(session_id)


def is_ending(session_id: str | None) -> bool:
    """True when a tool asked to end *session_id*'s current turn."""
    if not session_id:
        return False
    with _lock:
        return session_id in _ending


def clear(session_id: str) -> None:
    """Forget the request, at the start and end of every turn."""
    with _lock:
        _ending.discard(session_id)
