import autogpt_libs.auth as auth
from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel, ValidationError

from backend.api.features.experts import experts_db
from backend.copilot.tools.models import ExpertOnboardingResponse, ResponseType
from backend.data.db import query_raw_with_schema

router = APIRouter()

SKIP_MESSAGE = "Let's skip the setup questions for now."


class OnboardingMessage(BaseModel):
    session_id: str
    role: str
    content: str


class PendingExpertOnboarding(BaseModel):
    session_id: str


@router.get("/{expert_id}/onboarding", operation_id="get_expert_onboarding")
async def get_expert_onboarding(
    expert_id: str,
    user_id: str = Security(auth.get_user_id),
) -> PendingExpertOnboarding | None:
    expert = await experts_db.get_expert(user_id, expert_id)
    if expert is None or expert.is_archived:
        raise HTTPException(status_code=404, detail="Expert not found")
    # Every user message counts, since any reply settles the card in its own
    # chat; only cards, answers and skips need their text.
    messages = await query_raw_with_schema(
        """
        SELECT m."sessionId" AS session_id, m.role,
          CASE
            WHEN m.role = 'tool' OR m.content = $3
              OR m.content LIKE '**Here are my answers:**%'
            THEN m.content
            ELSE ''
          END AS content
        FROM {schema_prefix}"ChatMessage" m
        JOIN {schema_prefix}"ChatSession" s ON s.id = m."sessionId"
        WHERE s."userId" = $1 AND s."expertId" = $2
          AND (
            (m.role = 'tool' AND m.content LIKE '%"expert_onboarding"%')
            OR m.role = 'user'
          )
        ORDER BY m."createdAt", m.sequence
        """,
        user_id,
        expert_id,
        SKIP_MESSAGE,
        model=OnboardingMessage,
    )
    session_id = pending_onboarding(messages, expert_id)
    return PendingExpertOnboarding(session_id=session_id) if session_id else None


def pending_onboarding(messages: list[OnboardingMessage], expert_id: str) -> str | None:
    """The chat whose onboarding card is still waiting on the user, if any."""
    cards: list[tuple[str, ExpertOnboardingResponse]] = []
    for message in messages:
        if message.role == "tool":
            try:
                card = ExpertOnboardingResponse.model_validate_json(message.content)
            except ValidationError:
                continue
            if (
                card.type == ResponseType.EXPERT_ONBOARDING
                and card.expert_id == expert_id
                and card.steps
            ):
                cards.append((message.session_id, card))
        elif message.role == "user" and cards:
            if message.content == SKIP_MESSAGE or any(
                _answers_card(message.content, card) for _, card in cards
            ):
                return None
            # Mirrors the chat, where any reply settles the card it follows.
            cards = [entry for entry in cards if entry[0] != message.session_id]
    return cards[-1][0] if cards else None


def _answers_card(content: str, card: ExpertOnboardingResponse) -> bool:
    if not content.startswith("**Here are my answers:**\n\n"):
        return False
    if not content.endswith("\n\nPlease proceed."):
        return False
    remaining = content.removeprefix("**Here are my answers:**\n\n").removesuffix(
        "\n\nPlease proceed."
    )
    for index, step in enumerate(card.steps):
        question = f"> {step.question}\n\n"
        if not remaining.startswith(question):
            return False
        remaining = remaining.removeprefix(question)
        if index + 1 < len(card.steps):
            next_question = f"\n\n> {card.steps[index + 1].question}\n\n"
            answer, separator, remaining = remaining.partition(next_question)
            if not separator or not answer.strip():
                return False
            remaining = next_question.removeprefix("\n\n") + remaining
        elif not remaining.strip():
            return False
    return True
