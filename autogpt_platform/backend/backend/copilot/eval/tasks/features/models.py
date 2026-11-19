"""The feature eval's data: cases for task controls and suggested tasks,
and what a run of each produced.

A case names the simulated company it lives in (``data/worlds`` of the task
eval), states its expectation in plain words (``must`` / ``must_not``) and
lists the expectations code can check (``checks``). ``needs`` says what it
takes to run: nothing but code, a model, or the live database and services.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.copilot.eval.tasks.models import (
    CheckResult,
    Checks,
    HumanExample,
    Rating,
    Run,
)

Area = Literal["controls", "suggestions"]
Needs = Literal["none", "model", "live"]
LiveAction = Literal["start", "retry", "stop", "start_suggestion"]


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class TaskSpec(_Open):
    """A task as the user made it on the board."""

    title: str
    description: str | None = None
    goal: str | None = None
    done_when: str | None = None
    owner: str | None = Field(default=None, description="A world expert's id.")
    status: str = "TODO"
    result: str | None = None
    failure_reason: str | None = None
    attempt: int = 1
    pad_description_to: int | None = Field(
        default=None,
        description="Repeat the description up to this many characters: a long "
        "brief, such as pasted notes.",
    )


class BoardTask(_Open):
    title: str
    status: str = "DONE"
    goal: str | None = None
    result: str | None = None
    failure_reason: str | None = None
    status_text: str | None = None
    owner: str | None = None
    days_ago: float = 1


class AccountSpec(_Open):
    """What a user's account holds when Otto drafts their next tasks."""

    board: list[BoardTask] = Field(default_factory=list)
    live: list[BoardTask] = Field(
        default_factory=list, description="Suggestions not yet started or dismissed."
    )
    dismissed: list[str] = Field(default_factory=list)
    retired: list[str] = Field(
        default_factory=list,
        description="Suggestions retired unstarted lately, before this run.",
    )
    use_world_experts: bool = True


class FeatureChecks(_Open):
    """Machine-checked expectations; a field left out is not checked."""

    # Controls
    brief_has: list[str] = Field(
        default_factory=list,
        description="Parts of the task the worker's first message carries: "
        "title, goal, done_when, description, note.",
    )
    settles_every_ending: bool | None = None
    settles_as: list[str] = Field(default_factory=list)
    mentions_any: list[str] = Field(default_factory=list)
    fails_with_reason: bool | None = Field(
        default=None,
        description="A run that would settle FAILED has the worker's own reason, "
        "not the generic one.",
    )
    worker: Checks = Field(default_factory=Checks)
    # Suggestions
    model_called: bool | None = None
    min_suggestions: int | None = None
    max_suggestions: int | None = None
    max_new: int | None = None
    no_repeats: bool | None = None
    no_rewords: bool | None = None
    reasons_grounded: bool | None = None
    owners_valid: bool | None = None
    owner_for: dict[str, str] = Field(default_factory=dict)
    no_generic: bool | None = None
    briefs_actionable: bool | None = None
    has_done_when: bool | None = None
    forbidden: list[str] = Field(default_factory=list)
    # Live
    settled: bool | None = None
    attempt_up: bool | None = None
    note_on_record: bool | None = None
    nothing_after_stop: bool | None = None
    suggested_cleared: bool | None = None


class FeatureExpected(_Open):
    must: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    checks: FeatureChecks = Field(default_factory=FeatureChecks)


class FeatureCase(_Open):
    id: str
    area: Area
    world: str
    title: str
    needs: Needs = "none"
    review: bool = False
    expected: FeatureExpected = Field(default_factory=FeatureExpected)
    human_example: HumanExample = Field(default_factory=HumanExample)
    task: TaskSpec | None = None
    note: str | None = None
    account: AccountSpec | None = None
    scripted_reply: list[dict[str, Any]] | None = Field(
        default=None,
        description="Stands in for the drafting model, so the code's own "
        "guards are checked with no model call.",
    )
    live: LiveAction | None = None


class FeatureResult(BaseModel):
    """One case's outcome. ``ratings`` holds one per reviewer; a case with no
    reviewer judgement gets the same code-only rating under every name."""

    case_id: str
    area: Area
    title: str
    world: str
    needs: Needs
    skipped: str | None = None
    error: str | None = None
    checks: list[CheckResult] = Field(default_factory=list)
    ratings: dict[str, Rating] = Field(default_factory=dict)
    reviewed: bool = False
    output: dict[str, Any] = Field(default_factory=dict)
    run: Run | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
