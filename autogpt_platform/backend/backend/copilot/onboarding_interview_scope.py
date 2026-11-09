"""The onboarding interview's scope guard on the server: when off-topic turns
end an interview.

The interview's prompt holds the line itself (``onboarding_interview_goals``):
Otto redirects a request for a side task, and after two redirects sums up and
says goodbye. This module is the server's check on the same line, in two
parts:

- :func:`scope_verdict`, the rule, a count over per-turn labels: a turn about
  the person's work clears it, a request for a side task adds one, and a
  harmful request adds one and counts on its own too. Three requests in a row
  (the ask, then a push past each of two redirects) or two harmful ones (the
  ask, then a push past a refusal) close the interview.
- :func:`interview_scope_classifier`, the seam for a scope classifier: a
  cheap model call that labels the person's latest turns. It is off by
  default, and while it is off no turn is labelled: the prompt's own
  two-redirect rule and the reply cap bound the interview.

The judgement runs beside the turn, never in front of it, so it adds nothing
to the time to the first token. A close it calls for lands once that turn
has ended (``onboarding_dump.interview``).
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from backend.copilot.expert_kickoff import is_hidden_chat_message
from backend.copilot.onboarding_interview import InterviewEndReason

if TYPE_CHECKING:
    from backend.copilot.model import ChatSession

logger = logging.getLogger(__name__)

# What a user turn of an interview is about. "other" (a greeting, a yes or
# no, a question about the interview itself) leaves the count as it is.
ScopeLabel = Literal["work", "request", "harmful", "other"]

REQUESTS_IN_A_ROW_TO_CLOSE = 3
HARMFUL_TO_CLOSE = 2

# How many of the person's latest turns a classifier is shown: enough to see
# the run the rule counts.
SCOPE_WINDOW = REQUESTS_IN_A_ROW_TO_CLOSE

# How long a judgement may take before the turn is left unjudged.
CLASSIFIER_TIMEOUT_SECONDS = 8.0


class ScopedTurn(BaseModel):
    """One of the person's turns, with the line of Otto's it answered."""

    model_config = ConfigDict(frozen=True)

    otto: str
    user: str


class InterviewScopeClassifier(Protocol):
    """Labels the person's turns, oldest first, one label per turn."""

    async def __call__(self, turns: Sequence[ScopedTurn]) -> Sequence[ScopeLabel]: ...


def interview_scope_classifier() -> InterviewScopeClassifier | None:
    """The scope classifier interview turns are judged with; ``None`` while
    it is off, which it is by default.

    Turning it on means returning a classifier here: a cheap model call that
    labels each of the turns it is given (see :data:`ScopeLabel`).
    """
    return None


def scope_verdict(labels: Sequence[ScopeLabel]) -> InterviewEndReason | None:
    """``"scope"`` when the labels, oldest first, end the interview."""
    requests = 0
    harmful = 0
    for label in labels:
        if label == "work":
            requests = 0
            harmful = 0
        elif label in ("request", "harmful"):
            requests += 1
            if label == "harmful":
                harmful += 1
    if requests >= REQUESTS_IN_A_ROW_TO_CLOSE or harmful >= HARMFUL_TO_CLOSE:
        return "scope"
    return None


def recent_turns(session: "ChatSession") -> list[ScopedTurn]:
    """The person's latest turns, oldest first, each with the line of Otto's
    before it. Only rows the person wrote and Otto's visible text count:
    hidden rows, tool rows and empty replies are left out."""
    turns: list[ScopedTurn] = []
    otto = ""
    for row in session.messages:
        if row.role == "assistant" and (row.content or "").strip():
            otto = row.content or ""
        elif row.role == "user" and not is_hidden_chat_message(row.metadata):
            turns.append(ScopedTurn(otto=otto, user=row.content or ""))
    return turns[-SCOPE_WINDOW:]


async def judge_interview_scope(session: "ChatSession") -> InterviewEndReason | None:
    """Whether the person's latest turn, already in *session*, ends the
    interview on scope. ``None`` while the classifier is off, and whenever it
    fails or runs out of time: an outage never ends an interview."""
    classifier = interview_scope_classifier()
    if classifier is None:
        return None
    turns = recent_turns(session)
    if not turns:
        return None
    try:
        labels = await asyncio.wait_for(
            classifier(turns), timeout=CLASSIFIER_TIMEOUT_SECONDS
        )
    except Exception:
        logger.warning(
            f"Onboarding interview {session.session_id[:12]}: scope not judged",
            exc_info=True,
        )
        return None
    return scope_verdict(labels)
