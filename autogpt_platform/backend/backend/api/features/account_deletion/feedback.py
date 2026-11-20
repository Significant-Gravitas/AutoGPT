"""Anonymous exit feedback for self-serve account deletion (SECRT-2877)."""

from backend.api.features.account_deletion.analytics import plan_state, tenure_bucket
from backend.api.features.account_deletion.models import DeletionFeedbackRequest
from backend.api.features.account_deletion.rate_limit import enforce_feedback_rate_limit
from backend.data.user import get_user_by_id
from backend.util.posthog_client import capture_anonymous
from backend.util.posthog_events import PostHogEvent


async def record_feedback(user_id: str, body: DeletionFeedbackRequest) -> None:
    """Send one open of the Delete account dialog, one answer to "Why are you
    leaving?", or one click on the alternative it offered, as an anonymous
    PostHog event.

    The user id only keys the rate limit and loads the plan and account age,
    which are computed here so the client cannot set them. Nothing that
    identifies the person is sent. A value the person left out is left out of
    the event. The typed text is sent as is and never logged.
    """
    await enforce_feedback_rate_limit(user_id)
    user = await get_user_by_id(user_id)
    event, fields = _event(body)
    fields["plan_state"] = plan_state(user.subscription_tier)
    fields["tenure_bucket"] = tenure_bucket(user.created_at)
    capture_anonymous(
        event, {key: value for key, value in fields.items() if value is not None}
    )


def _event(body: DeletionFeedbackRequest) -> tuple[PostHogEvent, dict[str, str | None]]:
    if body.kind == "opened":
        return PostHogEvent.ACCOUNT_DELETION_STARTED, {"surface": body.surface}
    if body.kind == "alternative_clicked":
        return PostHogEvent.ACCOUNT_DELETION_ALTERNATIVE_CLICKED, {
            "reason": body.reason,
            "alternative": body.alternative,
        }
    return PostHogEvent.ACCOUNT_DELETION_FEEDBACK, {
        "reason": body.reason,
        "text": body.text,
    }
