"""The PostHog events feedback prompts send, for every user whatever their
cookie consent, which is why answers come here and not from the browser."""

from prisma.models import FeedbackPrompt

from backend.util.posthog_client import capture
from backend.util.posthog_events import PostHogEvent


def capture_prompt_shown(user_id: str, prompt_id: str, kind: str) -> None:
    capture(
        user_id,
        PostHogEvent.FEEDBACK_PROMPT_SHOWN,
        {"prompt_id": prompt_id, "kind": kind},
    )


def capture_prompt_answered(user_id: str, prompt: FeedbackPrompt) -> None:
    """Send the answer as it now stands. Each tap sends another event, and the
    latest per ``prompt_id`` is the answer that stands."""
    capture(
        user_id,
        PostHogEvent.FEEDBACK_PROMPT_ANSWERED,
        {
            "prompt_id": prompt.id,
            "kind": prompt.kind,
            "reason": prompt.reason,
            "note": prompt.note,
            "score": prompt.score,
            "offer_taken": prompt.offerTaken,
        },
    )
