"""Rating an AutoPilot reply: the thumbs up/down and copy actions under it.

Served under ``/api/chat`` next to the other session routes; the rating
logic lives in ``backend.copilot.feedback``.
"""

from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.copilot.feedback import (
    MAX_FEEDBACK_COMMENT_CHARS,
    FeedbackScoreName,
    FeedbackScoreValue,
    MessageFeedbackResponse,
    record_message_feedback,
)

router = APIRouter(tags=["chat"])


class MessageFeedbackRequest(BaseModel):
    """One rating of one reply."""

    message_id: str = Field(
        min_length=1,
        max_length=256,
        description=(
            "The reply as the chat UI knows it: ``<session id>-seq-<N>`` for "
            "a reply loaded from the chat history, or the message's own id."
        ),
    )
    score_name: FeedbackScoreName = Field(
        description='"user-feedback" for thumbs up/down, "copy" when the '
        "user copied the reply.",
    )
    score_value: FeedbackScoreValue = Field(
        description="1 for thumbs up or a copy, 0 for thumbs down.",
    )
    comment: str | None = Field(
        default=None,
        max_length=MAX_FEEDBACK_COMMENT_CHARS,
        description="What the user said was wrong, from the thumbs-down form.",
    )

    @field_validator("comment")
    @classmethod
    def blank_comment_is_none(cls, comment: str | None) -> str | None:
        if comment is None:
            return None
        return comment.strip() or None

    @model_validator(mode="after")
    def copy_scores_one(self) -> "MessageFeedbackRequest":
        if self.score_name == "copy" and self.score_value != 1:
            raise ValueError("A copy is always scored 1")
        return self


@router.post(
    "/sessions/{session_id}/feedback",
    summary="Submit message feedback",
    dependencies=[Security(auth.requires_user)],
    responses={
        404: {"description": "Session or reply not found, or not the caller's"},
    },
)
async def submit_message_feedback(
    session_id: str,
    request: MessageFeedbackRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> MessageFeedbackResponse:
    """Rate a reply in one of the caller's chats.

    Saves the rating, one per reply and score name so rating again
    overwrites it, and scores it in Langfuse on the trace of the turn that
    wrote the reply, or on the chat's Langfuse session when that trace is
    not known. Langfuse being down does not fail the request; the response
    says where the score went.
    """
    feedback = await record_message_feedback(
        user_id=user_id,
        session_id=session_id,
        message_id=request.message_id,
        score_name=request.score_name,
        score_value=request.score_value,
        comment=request.comment,
    )
    if feedback is None:
        raise HTTPException(
            status_code=404,
            detail=f"Message {request.message_id} not found or access denied",
        )
    return feedback
