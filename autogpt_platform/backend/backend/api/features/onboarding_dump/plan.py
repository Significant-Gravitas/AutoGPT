"""The plan: the greeting's suggested tasks ranked, each with an owner.

After the greeting and the team are stored, one TypeSafe Jev call scores how
much each task unblocks this person this week and picks who on the proposed
team should take it (or Otto). No new LLM call: Jev answers typed questions
in well under a second.

Same contract as ``copilot/gate/jev.py``: every failure returns None, and the
caller falls through to ``plan_owners.fallback_assign``.
"""

import asyncio
import logging
import os
from typing import Any

from pydantic import BaseModel, StrictFloat, StrictStr, ValidationError
from typesafe_sdk import Choice, Noul, Score

from backend.api.features.onboarding_dump.models import (
    RecommendedExpert,
    SuggestedPrompt,
)
from backend.blocks.typesafe._budget import prepare_state
from backend.blocks.typesafe._client import call_jev
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
_api_key = Settings().secrets.typesafe_jev_api_key
# None runs on the SDK's default Jev model.
_MODEL = os.environ.get("BRAIN_DUMP_PLAN_JEV_MODEL") or None
_TIMEOUT_SECONDS = 5

MAX_TRANSCRIPT_CHARS = 6_000
# The cut also stops at this many UTF-8 bytes: 6,000 characters of Japanese
# or Chinese (three bytes each) would not fit the request with the rest of
# the state, so a long dump in those scripts is cut shorter, not dropped.
MAX_TRANSCRIPT_BYTES = 12_000
# When the tasks themselves are long in a three-byte script, the transcript
# gives way further, in these steps, before the plan goes without Jev.
TRANSCRIPT_BYTE_STEPS = (MAX_TRANSCRIPT_BYTES, 6_000, 3_000)
# The owner question lists every candidate once per task, so a long team
# would spend the request on repeats: the recommendation's first six (it is
# ordered most useful first) are plenty for six tasks and a team of three.
MAX_OWNER_CANDIDATES = 6
# A title and the opening of the prompt say what the task is.
MAX_PROMPT_STATE_CHARS = 600
MAX_PROMPT_STATE_BYTES = 1_200
# A pick Jev is less sure of than this goes to Otto: a face the user meets on
# day one should be one the evidence put there.
OWNER_MIN_PROBABILITY = 0.4
OTTO = "otto"

RUBRIC = (
    "Rubric: a good first task removes the thing this person said costs "
    "them the most time; the team below can do it this week without "
    "accounts or inputs the person does not have yet; and it is concrete, "
    "with a result they can look at."
)
RANK_LEVELS = [
    "nothing they said points at it",
    "related to something they mentioned once",
    "addresses a named pain, but needs an account or input they do not have yet",
    "addresses a named pain and the team can start now",
    "addresses the pain they said costs them the most, and the team can start now",
]


class JudgedPlan(BaseModel):
    """Jev's ranking, with what the plan's log line reports about it.

    ``order`` is each ranked prompt's position in the greeting's list, so
    the rows keep the ids a reader may already have handed out.
    """

    prompts: list[SuggestedPrompt]
    order: list[int]
    scores: list[float]
    owner_probabilities: list[float]


class _RankAnswer(BaseModel):
    score: StrictFloat


class _OwnerAnswer(BaseModel):
    choice: StrictStr
    confidence: StrictFloat | None = None
    probabilities: dict[str, StrictFloat] = {}


class _JevTeammate(BaseModel):
    template_id: str
    name: str
    role: str
    workflow_names: list[str]


class _JevTask(BaseModel):
    index: int
    title: str
    prompt: str


class _JevState(BaseModel):
    """The rubric first, then everything Jev judges against;
    ``prepare_state`` serializes it in this order."""

    rubric: str
    role: str
    pain_points: list[str]
    transcript: str
    team: list[_JevTeammate]
    tasks: list[_JevTask]


async def judge_plan(
    transcript: str,
    user_role: str | None,
    pain_points: list[str],
    prompts: list[SuggestedPrompt],
    team: list[RecommendedExpert],
) -> JudgedPlan | None:
    """``prompts`` ordered by Jev's rank, each with its owner, or None.

    None for an unset key, a timeout, a raise, a cut state, and for any
    answer missing: a half-read plan is no plan.
    """
    if not _api_key or not prompts:
        return None
    candidates = team[:MAX_OWNER_CANDIDATES]
    questions = _questions(prompts, candidates)
    try:
        # A cut state would be judged on its head, so it is no answer.
        state = _fitting_state(
            transcript, user_role, pain_points, prompts, candidates, questions
        )
        if state is None:
            logger.warning("Brain dump plan: Jev state too long; using the fallback")
            return None
        result = await asyncio.wait_for(
            call_jev(
                _api_key, state, questions, model=_MODEL, timeout=_TIMEOUT_SECONDS
            ),
            timeout=_TIMEOUT_SECONDS + 0.5,
        )
    except TimeoutError:
        logger.warning("Brain dump plan: Jev timed out; using the fallback")
        return None
    except Exception as e:
        logger.warning(f"Brain dump plan: Jev raised ({type(e).__name__}); fallback")
        return None
    # ``error`` also reports soft problems (no request id, no usage) on a
    # complete answer, so only missing or cut answers count, as in jev.py.
    if not result.answers or result.truncated:
        logger.warning(
            f"Brain dump plan: Jev failed ({result.error or 'state cut'}); fallback"
        )
        return None
    judged = _read(result.answers, prompts, candidates)
    if judged is None:
        logger.warning("Brain dump plan: Jev answered an unusable body; fallback")
    return judged


def _fitting_state(
    transcript: str,
    user_role: str | None,
    pain_points: list[str],
    prompts: list[SuggestedPrompt],
    team: list[RecommendedExpert],
    questions: dict[str, Choice | Score | Noul],
) -> dict[str, Any] | None:
    """The state with as much of the transcript as fits the request."""
    for max_bytes in TRANSCRIPT_BYTE_STEPS:
        state = _state(
            clip_transcript(transcript, max_bytes),
            user_role,
            pain_points,
            prompts,
            team,
        )
        if not prepare_state(state, questions).truncated:
            return state
    return None


def _state(
    transcript: str,
    user_role: str | None,
    pain_points: list[str],
    prompts: list[SuggestedPrompt],
    team: list[RecommendedExpert],
) -> dict[str, Any]:
    return _JevState(
        rubric=RUBRIC,
        role=user_role or "",
        pain_points=pain_points,
        transcript=transcript,
        team=[
            _JevTeammate(
                template_id=expert.template_id,
                name=expert.name,
                role=expert.role,
                workflow_names=expert.workflow_names,
            )
            for expert in team
        ],
        tasks=[
            _JevTask(
                index=index,
                title=prompt.title,
                prompt=_clip(
                    prompt.prompt, MAX_PROMPT_STATE_CHARS, MAX_PROMPT_STATE_BYTES
                ),
            )
            for index, prompt in enumerate(prompts)
        ],
    ).model_dump()


def clip_transcript(transcript: str, max_bytes: int = MAX_TRANSCRIPT_BYTES) -> str:
    """The transcript's opening, at most ``MAX_TRANSCRIPT_CHARS`` characters
    and ``max_bytes`` UTF-8 bytes."""
    return _clip(transcript.strip(), MAX_TRANSCRIPT_CHARS, max_bytes)


def _clip(text: str, max_chars: int, max_bytes: int) -> str:
    # Cut on bytes, then drop a character split by the cut.
    return text[:max_chars].encode()[:max_bytes].decode(errors="ignore")


def _questions(
    prompts: list[SuggestedPrompt], team: list[RecommendedExpert]
) -> dict[str, Choice | Score | Noul]:
    owners: dict[str, Any] = {
        expert.template_id: _owner_label(expert) for expert in team
    }
    owners[OTTO] = "Nobody on this team fits; Otto does it himself"
    # Each task's rank, then its owner, as Jev reads them.
    return {
        key: question
        for index in range(len(prompts))
        for key, question in _task_questions(index, owners)
    }


def _task_questions(
    index: int, owners: dict[str, Any]
) -> tuple[tuple[str, Choice | Score | Noul], ...]:
    return (
        (
            f"rank_{index}",
            Score(
                instructions=f"How much does starting with task {index} unblock"
                " this person this week, by the rubric?",
                criteria=RANK_LEVELS,
            ),
        ),
        (
            f"owner_{index}",
            Choice(instructions=f"Who should take task {index}?", criteria=owners),
        ),
    )


def _owner_label(expert: RecommendedExpert) -> str:
    workflows = ", ".join(expert.workflow_names) or "no workflows listed"
    return f"{expert.name}, {expert.role}: {workflows}"


def _read(
    answers: dict[str, Any],
    prompts: list[SuggestedPrompt],
    team: list[RecommendedExpert],
) -> JudgedPlan | None:
    """Order by score (ties keep the greeting's order), owner by choice.

    A body that is not the shape asked for is no answer, like a missing one.
    """
    labels = {expert.template_id for expert in team} | {OTTO}
    try:
        ranks = [
            _RankAnswer.model_validate(answers.get(f"rank_{index}"))
            for index in range(len(prompts))
        ]
        owners = [
            _OwnerAnswer.model_validate(answers.get(f"owner_{index}"))
            for index in range(len(prompts))
        ]
    except ValidationError:
        return None
    if any(owner.choice not in labels for owner in owners):
        return None
    picks = [_pick(owner) for owner in owners]
    if any(probability is None for _, probability in picks):
        return None
    scores = [rank.score for rank in ranks]
    order = sorted(range(len(prompts)), key=lambda index: -scores[index])
    return JudgedPlan(
        prompts=[
            prompts[index].model_copy(update={"expert_template_id": picks[index][0]})
            for index in order
        ],
        order=order,
        scores=[scores[index] for index in order],
        owner_probabilities=[picks[index][1] or 0.0 for index in order],
    )


def _pick(owner: "_OwnerAnswer") -> tuple[str | None, float | None]:
    """The owner if Jev is sure enough of them, and how sure it is."""
    probability = owner.probabilities.get(owner.choice, owner.confidence)
    if probability is None:
        return None, None
    confident = owner.choice != OTTO and probability >= OWNER_MIN_PROBABILITY
    return (owner.choice if confident else None), probability
