"""Response models of the admin Evals page: the saved runs of the task eval
(``backend/copilot/eval/tasks``) and of its feature eval (``.../features``)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

EvalSuite = Literal["task", "features"]
EvalSource = Literal["baseline", "local"]
EvalVerdict = Literal["pass", "partial", "fail"]
EvalCaseCoverage = Literal["all", "not_passed", "none"]


class EvalStats(BaseModel):
    cases: int
    score: float = Field(description="Mean score, 0-100.")
    pass_rate: float = Field(description="Share of cases passed, 0-100.")
    passed: int
    partial: int
    failed: int


class EvalReviewerStats(BaseModel):
    reviewer: str = Field(description="As the run names it, e.g. claude:<model>.")
    label: str
    stats: EvalStats


class EvalRunSummary(BaseModel):
    id: str
    suite: EvalSuite
    source: EvalSource
    label: str
    started_at: datetime | None = None
    backend: str | None = None
    otto_model: str | None = None
    user_model: str | None = None
    reviewers: list[str] = Field(default_factory=list)
    stats: EvalStats = Field(description="The first reviewer's figures.")
    case_coverage: EvalCaseCoverage = Field(
        description="Which cases the saved files keep: every case, only the "
        "ones not passed by every reviewer, or none (totals only)."
    )
    series: str = Field(description="Runs of the same setup share a series.")


class EvalCommand(BaseModel):
    label: str
    command: str
    cost: str


class EvalRunList(BaseModel):
    runs: list[EvalRunSummary]
    commands: list[EvalCommand]


class EvalCategoryScore(BaseModel):
    name: str
    stats: EvalStats
    reviewers: list[EvalReviewerStats] = Field(default_factory=list)


class EvalRating(BaseModel):
    reviewer: str
    label: str
    verdict: EvalVerdict
    score: float
    reasons: list[str] = Field(default_factory=list)


class EvalCheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class EvalToolCall(BaseModel):
    name: str
    service: str | None = None
    arguments: str = Field(default="", description="JSON, cut to a preview.")
    refused: str | None = None
    held: bool = False
    wrote: bool = False
    failed: bool = False
    result_preview: str = ""


class EvalQuestion(BaseModel):
    question: str
    reason: str | None = None
    default_guess: str | None = None


class EvalTurn(BaseModel):
    user: str = ""
    reply: str = ""
    final_reply: str = ""
    ended_by: str = "reply"
    seconds: float = 0
    tool_calls: list[EvalToolCall] = Field(default_factory=list)
    questions: list[EvalQuestion] = Field(default_factory=list)


class EvalExpectation(BaseModel):
    behaviour: str | None = None
    must: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    human_reply: str = ""
    human_actions: list[str] = Field(default_factory=list)
    human_why: str = ""


class EvalCase(BaseModel):
    case_id: str
    category: str
    title: str = ""
    world: str = ""
    verdict: EvalVerdict | None = Field(
        default=None, description="The first reviewer's; none when skipped."
    )
    score: float | None = None
    previous_verdict: EvalVerdict | None = None
    request: str = ""
    expected: EvalExpectation | None = None
    observed: str | None = Field(
        default=None,
        description="What Otto did in the first turn: reply, start_task, ask, "
        "sign_in, delegate or task_update.",
    )
    ratings: list[EvalRating] = Field(default_factory=list)
    checks: list[EvalCheck] = Field(default_factory=list)
    turns: list[EvalTurn] = Field(default_factory=list)
    error: str | None = None
    skipped: str | None = None
    needs: str | None = None
    output: str | None = Field(default=None, description="Feature output, JSON.")
    detail: Literal["full", "report"] = Field(
        default="full",
        description="report: only what the run's report.md kept (verdicts, "
        "failed checks, two reasons per reviewer).",
    )


class EvalCaseChange(BaseModel):
    case_id: str
    title: str = ""
    category: str
    before: EvalVerdict
    after: EvalVerdict


class EvalCategoryChange(BaseModel):
    name: str
    pass_rate: float
    previous_pass_rate: float | None = None
    pass_rate_delta: float | None = None
    score_delta: float | None = None


class EvalComparison(BaseModel):
    previous_id: str
    previous_label: str
    pass_rate_delta: float
    score_delta: float
    failed_delta: int
    categories: list[EvalCategoryChange] = Field(default_factory=list)
    cases_compared: bool = False
    improved: list[EvalCaseChange] = Field(default_factory=list)
    regressed: list[EvalCaseChange] = Field(default_factory=list)


class EvalHistoryPoint(BaseModel):
    id: str
    label: str
    pass_rate: float
    score: float
    current: bool = False


class EvalRunDetail(BaseModel):
    run: EvalRunSummary
    reviewers: list[EvalReviewerStats]
    categories: list[EvalCategoryScore]
    cases: list[EvalCase]
    comparison: EvalComparison | None = None
    history: list[EvalHistoryPoint] = Field(default_factory=list)
    commands: list[EvalCommand] = Field(default_factory=list)
    seconds_per_turn: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    note: str | None = None
