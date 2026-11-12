"""EndInterviewTool: Otto's end signal in an onboarding interview."""

import copy
from typing import Any

from backend.copilot.model import (
    INTERVIEW_STAGE_CLOSED,
    INTERVIEW_STAGE_HANDED_OFF,
    ChatSession,
)
from backend.copilot.onboarding_interview import (
    clean_first_task,
    close_on_end_signal,
    is_onboarding_interview,
)
from backend.copilot.onboarding_interview_goals import (
    END_SIGNAL_FIRST_TASK,
    END_SIGNAL_PARAMETERS,
)

from .base import BaseTool
from .models import ErrorResponse, InterviewEndedResponse, ToolResponseBase

END_INTERVIEW_TOOL = "end_interview"


class EndInterviewTool(BaseTool):
    """Close the onboarding interview this chat is, on the first task the
    person agreed their team starts on.

    Offered only in an onboarding interview: ``session_permissions`` allows it
    there, ``tools.interview_disabled_tools`` hides it everywhere else, and a
    call from any other chat is refused here. The close is
    ``onboarding_interview.close_on_end_signal`` on whichever engine runs the
    turn: the stage moves to ``closed`` with the task stored beside it, and
    the API side finishes the close once the turn is over
    (``onboarding_dump.interview.close_after_turn``). The baseline engine ends
    the turn at the round this call is in.
    """

    @property
    def name(self) -> str:
        return END_INTERVIEW_TOOL

    @property
    def description(self) -> str:
        return (
            "End the interview once they've agreed to the first thing their "
            "team will start on. Call it in the same reply as your goodbye."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return copy.deepcopy(END_SIGNAL_PARAMETERS)

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        if user_id is None or not is_onboarding_interview(session):
            return ErrorResponse(
                message=(
                    f"{self.name} ends an onboarding interview, and this chat "
                    "is not one."
                ),
                error="not_an_onboarding_interview",
                session_id=session.session_id,
            )
        first_task = clean_first_task(kwargs.get(END_SIGNAL_FIRST_TASK))
        if not first_task:
            return ErrorResponse(
                message=(
                    f"Call {self.name} with {END_SIGNAL_FIRST_TASK}: the first "
                    "thing their team will start on, as they agreed it."
                ),
                error="first_task_missing",
                session_id=session.session_id,
            )
        if await close_on_end_signal(session.session_id, user_id, first_task):
            session.metadata.onboarding_interview_first_task = first_task
        # The turn's own copy reads as closed from here on, as the stored one
        # does whichever call closed it: the engines end the turn on it.
        if session.metadata.onboarding_interview_stage != INTERVIEW_STAGE_HANDED_OFF:
            session.metadata.onboarding_interview_stage = INTERVIEW_STAGE_CLOSED
        return InterviewEndedResponse(
            message=(
                "The interview is closed, and their team starts on first_task "
                "first. If you have not said goodbye yet, say a short goodbye "
                "now, in their language, that tells them so. Ask nothing."
            ),
            first_task=first_task,
            session_id=session.session_id,
        )
