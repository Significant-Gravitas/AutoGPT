"""Triggering heuristic for the planner phase.

Keeps the "is this worth planning?" decision in a single swappable helper so a
future cheap-classifier call (e.g. ``fast_standard_model``) can replace the
keyword/length heuristic without touching the baseline path.
"""

from __future__ import annotations

# Words that strongly suggest a build/task request rather than a
# conversational message. Lower-cased substring match.
_MULTISTEP_KEYWORDS = (
    "build",
    "create",
    "make me",
    "implement",
    "set up",
    "setup",
    "automate",
    "automation",
    "workflow",
    "pipeline",
    "agent",
    "scrape",
    "integrate",
    "schedule",
    "generate",
    "research",
    "step 1",
    "step 2",
    "and then",
    "then ",
    "first,",
    "after that",
)

# Below this length a message is almost always conversational; skip the planner
# to avoid paying for an expensive planning call on "hi" / "thanks".
_MIN_MULTISTEP_CHARS = 40


def is_multi_step_request(message: str | None) -> bool:
    """Cheap heuristic: does this request look like a multi-step task?

    v1 rule: the message must clear a minimum length AND contain a task-like
    keyword.  Deliberately conservative — a false negative just means the turn
    runs on the normal single-loop path (no regression), while a false positive
    pays for an unnecessary planner call.
    """
    text = (message or "").strip().lower()
    if len(text) < _MIN_MULTISTEP_CHARS:
        return False
    return any(kw in text for kw in _MULTISTEP_KEYWORDS)
