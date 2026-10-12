"""Placeholder: the feedback emitter is written in the next stage."""

from backend.api.features.account_deletion.models import DeletionFeedbackRequest


async def record_feedback(user_id: str, body: DeletionFeedbackRequest) -> None:
    raise NotImplementedError
