"""Tools that close the turn, and what may still happen after them.

A question for the user (``ask_question``) is the last thing in its turn: no
more text and no more tool calls, so the question card is what the user is
left looking at. Each engine then makes no further model call (the SDK
through its PostToolUse hook, the baseline by leaving its tool loop) and
refuses every tool call that starts later; calls the model placed before the
question in the same step have already run.

A background worker that leaves its task's result (``finish_task``) ends its
turn the same way: the summary and the document are the result, and nobody
is waiting on a closing message in the worker's chat.

Starting a background task (``start_task``) wraps the turn up instead: the
work runs elsewhere, so the turn may still start more tasks and say one
closing line, but every other tool call is refused, so none of the task is
done inline and the chat is free again at once.

The registry also counts the tool calls each turn has started, so a question
can tell whether anything was looked at before it (``question_rules``).

Process-local: the tool runs in the same executor process as the engine that
checks. Each turn clears its session when it starts, and the executor clears
it again when the turn is over, so an entry never reaches a later turn.
"""

import threading
from typing import Literal

ENDED_REFUSAL = (
    "Not run: you asked the user a question, which ends your turn. "
    "Call this again after they answer."
)
FINISHED_REFUSAL = (
    "Not run: the task is finished and its result is saved, which ends your turn."
)
WRAP_UP_REFUSAL = (
    "Not run: you started this as a background task, so the worker does it. "
    "If you have not told the user it started, say so in one short line; "
    "then stop."
)
_STILL_ALLOWED_AFTER_TASK = frozenset({"start_task"})

_lock = threading.Lock()
_state: dict[str, Literal["question", "task", "finished"]] = {}
_calls: dict[str, int] = {}


def note_call(session_id: str | None) -> None:
    """A tool call is starting in *session_id*'s turn."""
    if not session_id:
        return
    with _lock:
        _calls[session_id] = _calls.get(session_id, 0) + 1


def calls_started(session_id: str | None) -> int:
    """How many tool calls *session_id*'s turn has started, this one included."""
    if not session_id:
        return 0
    with _lock:
        return _calls.get(session_id, 0)


def request_end(session_id: str) -> None:
    """A question was asked: end *session_id*'s turn after this step."""
    with _lock:
        _state[session_id] = "question"


def request_finish(session_id: str) -> None:
    """A worker left its task's result: end *session_id*'s turn after this step."""
    with _lock:
        _state.setdefault(session_id, "finished")


def request_wrap_up(session_id: str) -> None:
    """A task was started: only more tasks and a closing line may follow."""
    with _lock:
        _state.setdefault(session_id, "task")


def is_ending(session_id: str | None) -> bool:
    """True when no further model call may run: the turn asked a question, or
    a worker finished its task."""
    return _get(session_id) in ("question", "finished")


def refusal(session_id: str | None, tool_name: str) -> str | None:
    """Why a call to *tool_name* may not start now, or None when it may."""
    state = _get(session_id)
    if state == "question":
        return ENDED_REFUSAL
    if state == "finished":
        return FINISHED_REFUSAL
    if state == "task" and tool_name not in _STILL_ALLOWED_AFTER_TASK:
        return WRAP_UP_REFUSAL
    return None


def clear(session_id: str) -> None:
    """Forget the request, at the start and end of every turn."""
    with _lock:
        _state.pop(session_id, None)
        _calls.pop(session_id, None)


def _get(session_id: str | None) -> Literal["question", "task", "finished"] | None:
    if not session_id:
        return None
    with _lock:
        return _state.get(session_id)
