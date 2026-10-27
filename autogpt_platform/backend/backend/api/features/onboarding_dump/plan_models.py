"""The plan step's response and the hire that follows the trial."""

from typing import Literal

from pydantic import BaseModel

from backend.api.features.onboarding_dump.models import RecommendedExpert, SuggestedTask


class PlanResponse(BaseModel):
    """The plan step's content: what Otto heard, the team, and the tasks.

    ``ready`` is false only while the background jobs are still running;
    the tasks and team it carries then are the role plan, so a client that
    stops waiting has something true to show. ``source`` is "llm" for a
    plan built from the tasks the greeting wrote (ranked by Jev or by the
    deterministic fallback) and "fallback" for a default set: the role
    sets, or the generic starters.
    """

    ready: bool
    source: Literal["llm", "fallback"]
    skipped_dump: bool
    heard: str | None = None
    team: list[RecommendedExpert] = []
    tasks: list[SuggestedTask] = []


class PlanHiredExpert(BaseModel):
    template_id: str
    expert_id: str
    name: str


class PlanHireFailure(BaseModel):
    template_id: str
    name: str


class PlanHireResponse(BaseModel):
    """The plan's team after a hire: everyone now on the user's team, and
    whoever could not join (a failed hire, or the active expert limit)."""

    hired: list[PlanHiredExpert] = []
    failed: list[PlanHireFailure] = []
