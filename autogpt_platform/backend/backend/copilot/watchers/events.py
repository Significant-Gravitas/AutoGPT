"""What the proactive "I noticed X" watchers actually say.

Pure text and ids — no DB, no Redis, no flags. Delivery lives in
:mod:`backend.copilot.watchers.deliver`.

The copy obeys the same three rules as ``executor.expert_posts``, for the
same reasons:

* **The opener says it happened autonomously.** These messages answer no
  question the user asked. Without "I noticed … while running on its
  schedule" they read as a reply to nothing.
* **Untrusted run text is quoted and attributed.** Errors and review
  instructions come out of workflow execution; replaying them in the
  expert's own voice would let scraped "ignore previous instructions"
  content read as assistant speech in the thread's history. They are
  blockquoted, introduced by who said them, and length-capped so one bad
  run can't persist a multi-MB message that reloads every later turn.
* **One short preamble sentence.** The card preview in the thread list is
  clamped to two lines; anything longer is invisible where it matters.
"""

import uuid
from enum import Enum

from prisma.enums import TriggerSource

# Discriminator the thread renderer keys on. Distinct from
# ``expert_posts.RUN_METADATA_KIND`` because these are attention cards, not
# completed-work cards; a client that doesn't know the kind falls back to
# rendering the markdown, which reads fine on its own.
WATCHER_METADATA_KIND = "copilot_watcher"

# Fixed namespace so the same event always derives the same message id.
# Redelivery — an executor retry, a re-fired pause check, a review row
# re-read — collides on ChatMessage's primary key, and that PK uniqueness
# *is* the dedupe. No "have I sent this already?" bookkeeping to get wrong.
_WATCHER_NAMESPACE = uuid.UUID("3c9f4a17-5b28-4d6e-8a13-6f0e2d7b95c4")

MAX_QUOTED_LENGTH = 500


class WatcherEvent(str, Enum):
    RUN_FAILED = "run_failed"
    EXPERT_PAUSED = "expert_paused"
    REVIEW_WAITING = "review_waiting"


# Why the run was going, in the expert's voice. Keep these as trailing
# clauses so they slot into a sentence without re-punctuating it.
_TRIGGER_CLAUSE: dict[TriggerSource, str] = {
    TriggerSource.cron: "while running on its schedule",
    TriggerSource.webhook: "while running from one of your triggers",
    TriggerSource.manual: "while running the job you started",
    TriggerSource.delegated: "while running work I handed off",
}


def trigger_clause(trigger_source: TriggerSource) -> str:
    return _TRIGGER_CLAUSE.get(trigger_source, _TRIGGER_CLAUSE[TriggerSource.manual])


def watcher_message_id(event: WatcherEvent, dedupe_key: str) -> str:
    return str(uuid.uuid5(_WATCHER_NAMESPACE, f"{event.value}:{dedupe_key}"))


def truncate(text: str, limit: int = MAX_QUOTED_LENGTH) -> str:
    return text if len(text) <= limit else f"{text[:limit]}… (truncated)"


def quote_lines(text: str) -> str:
    return "\n".join(f"> {line}" for line in text.splitlines() or [""])


def run_link(library_agent_id: str | None) -> str:
    return (
        f"\n\n[View the run](/library/agents/{library_agent_id})"
        if library_agent_id
        else ""
    )


def build_run_failed_message(
    agent_name: str,
    trigger_source: TriggerSource,
    error: str | None = None,
    library_agent_id: str | None = None,
) -> str:
    detail = (
        f"\n\nThe error it reported:\n\n{quote_lines(truncate(error))}" if error else ""
    )
    return (
        f"I noticed **{agent_name}** failed {trigger_clause(trigger_source)}."
        f"{detail}\n\n"
        f"Want me to run it again, or would you rather check its setup first?"
        f"{run_link(library_agent_id)}"
    )


def build_expert_paused_message(spent: int, budget: int) -> str:
    """No trigger clause here: the pause is not a run, and its "why" is the
    budget itself, which the sentence already states."""
    return (
        f"I noticed I've paused myself — I've used {spent} of my {budget} "
        "weekly credits, so my scheduled runs and triggers have stopped.\n\n"
        "Raise my budget or resume me from the Team page and I'll pick "
        "straight back up."
    )


def build_review_waiting_message(
    agent_name: str,
    trigger_source: TriggerSource,
    instructions: str | None = None,
    library_agent_id: str | None = None,
) -> str:
    asked = (
        f"\n\nWhat it's waiting on:\n\n{quote_lines(truncate(instructions))}"
        if instructions
        else ""
    )
    return (
        f"I noticed **{agent_name}** stopped for your approval "
        f"{trigger_clause(trigger_source)}."
        f"{asked}\n\n"
        f"Approve or reject it and I'll carry on from there."
        f"{run_link(library_agent_id)}"
    )
