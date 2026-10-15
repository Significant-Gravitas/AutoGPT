import autogpt_libs.auth as auth
from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel, ValidationError

from backend.api.features.experts import experts_db
from backend.copilot.tools.models import ExpertOnboardingResponse, ResponseType
from backend.data.db import query_raw_with_schema

router = APIRouter()

SKIP_MESSAGE = "Let's skip the setup questions for now."


class OnboardingMessage(BaseModel):
    role: str
    content: str


@router.get("/{expert_id}/onboarding", operation_id="get_expert_onboarding")
async def get_expert_onboarding(
    expert_id: str,
    user_id: str = Security(auth.get_user_id),
) -> ExpertOnboardingResponse | None:
    expert = await experts_db.get_expert(user_id, expert_id)
    if expert is None or expert.is_archived:
        raise HTTPException(status_code=404, detail="Expert not found")
    messages = await query_raw_with_schema(
        """
        SELECT m.role, m.content
        FROM {schema_prefix}"ChatMessage" m
        JOIN {schema_prefix}"ChatSession" s ON s.id = m."sessionId"
        WHERE s."userId" = $1 AND s."expertId" = $2
          AND (
            (m.role = 'tool' AND m.content LIKE '%"expert_onboarding"%')
            OR (m.role = 'user' AND (
              m.content = $3 OR m.content LIKE '**Here are my answers:**%'
            ))
          )
        ORDER BY m."createdAt", m.sequence
        """,
        user_id,
        expert_id,
        SKIP_MESSAGE,
        model=OnboardingMessage,
    )
    return pending_onboarding(messages, expert_id)


def pending_onboarding(
    messages: list[OnboardingMessage], expert_id: str
) -> ExpertOnboardingResponse | None:
    cards: list[ExpertOnboardingResponse] = []
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
                cards.append(card)
        elif message.role == "user" and cards:
            if message.content == SKIP_MESSAGE or any(
                _answers_card(message.content, card) for card in cards
            ):
                return None
    return cards[-1] if cards else None


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
