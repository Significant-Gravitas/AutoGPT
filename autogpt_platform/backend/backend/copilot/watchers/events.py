import uuid
from enum import Enum
from urllib.parse import quote

WATCHER_METADATA_KIND = "copilot_watcher"
MAX_QUOTED_LENGTH = 500

_WATCHER_NAMESPACE = uuid.UUID("3c9f4a17-5b28-4d6e-8a13-6f0e2d7b95c4")


class WatcherEvent(str, Enum):
    RUN_FAILED = "run_failed"
    EXPERT_PAUSED = "expert_paused"
    REVIEW_WAITING = "review_waiting"
    OVERFLOW = "overflow"


_TRIGGER_LABEL = {
    "cron": "on schedule",
    "webhook": "from a trigger",
    "manual": "on the requested run",
    "delegated": "on delegated work",
}


def watcher_message_id(event: WatcherEvent, dedupe_key: str) -> str:
    return str(uuid.uuid5(_WATCHER_NAMESPACE, f"{event.value}:{dedupe_key}"))


def truncate(text: str, limit: int = MAX_QUOTED_LENGTH) -> str:
    return text if len(text) <= limit else f"{text[:limit]}…"


def quote_lines(text: str) -> str:
    return "\n".join(f"> {line}" for line in text.splitlines() or [""])


def run_href(library_agent_id: str | None, execution_id: str) -> str:
    if library_agent_id is None:
        return "/home"
    return (
        f"/library/agents/{quote(library_agent_id, safe='')}"
        f"?activeTab=runs&activeItem={quote(execution_id, safe='')}"
    )


def build_run_failed_message(
    agent_name: str,
    trigger_source: str,
    error: str | None,
) -> str:
    detail = f"\n\nReported issue:\n\n{quote_lines(truncate(error))}" if error else ""
    trigger = _TRIGGER_LABEL.get(trigger_source, "during a workflow run")
    return f"**{agent_name} needs attention**\n\nThe workflow failed {trigger}.{detail}"


def build_expert_paused_message(expert_name: str) -> str:
    return (
        f"**{expert_name} needs your decision**\n\n"
        "Scheduled work is paused because the weekly limit was reached. "
        "Review or raise the limit from Team to continue."
    )


def build_review_waiting_message(agent_name: str, instructions: str | None) -> str:
    detail = (
        f"\n\nDecision requested:\n\n{quote_lines(truncate(instructions))}"
        if instructions
        else ""
    )
    return f"**{agent_name} needs your approval**{detail}"


def build_overflow_message() -> str:
    return (
        "**More expert updates need attention**\n\n"
        "Several workflows changed after today's notification limit. "
        "Open Home for the current status and next decisions."
    )
