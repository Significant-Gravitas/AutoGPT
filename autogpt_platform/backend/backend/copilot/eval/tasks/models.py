"""The task eval's data: simulated worlds, cases, and what a run produced."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.copilot.autopilot_mode import AutopilotMode

Behaviour = Literal[
    "reply", "start_task", "ask", "sign_in", "refuse", "delegate", "task_update"
]
Verdict = Literal["pass", "partial", "fail"]


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class Account(_Open):
    provider: str
    account: str = ""


class WorldTask(_Open):
    id: str
    title: str
    status: str = "IN_PROGRESS"
    owner: str | None = None


class World(_Open):
    """One simulated company and the person who talks to Otto."""

    id: str
    company: dict[str, Any] = Field(default_factory=dict)
    user: dict[str, Any] = Field(default_factory=dict)
    team: list[dict[str, Any]] = Field(default_factory=list)
    approval_mode: AutopilotMode = "auto"
    connected_accounts: list[Account] = Field(default_factory=list)
    not_connected: list[str] = Field(default_factory=list)
    memories: list[str] = Field(default_factory=list)
    business_understanding: str = ""
    experts: list[dict[str, Any]] = Field(default_factory=list)
    tasks: list[WorldTask] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    web: dict[str, Any] = Field(default_factory=dict)

    @property
    def connected(self) -> set[str]:
        return {a.provider.lower() for a in self.connected_accounts}


class Turn(_Open):
    user: str


class Answer(_Open):
    topic: str
    answer: str


class Checks(_Open):
    """Machine-checked expectations; a field left out is not checked."""

    max_tool_calls: int | None = None
    tools_called: list[str] = Field(default_factory=list)
    tools_not_called: list[str] = Field(default_factory=list)
    asks_question: bool | None = None
    max_questions: int | None = None
    question_reasons: list[str] = Field(default_factory=list)
    ends_with_question: bool | None = None
    services_called: list[str] = Field(default_factory=list)
    services_not_called: list[str] = Field(default_factory=list)
    no_writes: bool | None = Field(
        default=None,
        description="True: nothing is sent, posted or changed in any service "
        "(a write held for approval does not count).",
    )
    no_writes_first_turn: bool | None = Field(
        default=None,
        description="True: the first turn sends, posts or changes nothing (it "
        "should ask first); later turns may act on the answer.",
    )
    delegates: bool | None = Field(
        default=None,
        description="Hands the work to a hired expert: delegate_to_expert, or "
        "start_task / create_task with their expert_id.",
    )


class Expected(_Open):
    behaviour: Behaviour | str = "reply"
    must: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    checks: Checks = Field(default_factory=Checks)


class HumanExample(_Open):
    reply: str = ""
    actions: list[str] = Field(default_factory=list)
    why: str = ""


class Case(_Open):
    id: str
    category: str
    world: str
    title: str = ""
    approval_mode: AutopilotMode | None = None
    turns: list[Turn]
    answer_sheet: list[Answer] = Field(default_factory=list)
    default_answer: str = "use your best guess"
    min_accounts: list[str] = Field(default_factory=list)
    expected: Expected = Field(default_factory=Expected)
    human_example: HumanExample = Field(default_factory=HumanExample)


class ToolCall(BaseModel):
    """One call as the run recorded it: ``name`` is the tool that ran, with
    ``run_capability(id="tool:x")`` recorded as ``x``."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    service: str | None = None
    refused: str | None = None
    held: bool = Field(
        default=False,
        description="Held for the user's approval by the simulated gate; did not run.",
    )
    wrote: bool = Field(
        default=False, description="A send, post or change that ran in a service."
    )
    result_preview: str = ""


class Question(BaseModel):
    question: str
    reason: str | None = None
    default_guess: str | None = None


class TurnRecord(BaseModel):
    user: str
    reply: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    ended_by: Literal["reply", "question", "task_started", "round_cap", "error"] = (
        "reply"
    )
    rounds: int = 0
    seconds: float = 0.0


class Run(BaseModel):
    case_id: str
    turns: list[TurnRecord] = Field(default_factory=list)
    writes: list[dict[str, Any]] = Field(
        default_factory=list, description="Writes that ran (send, post, create...)."
    )
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class Review(BaseModel):
    verdict: Verdict
    score: float = Field(ge=0, le=10)
    reasons: list[str] = Field(default_factory=list)


class Rating(BaseModel):
    """One reviewer's rating of a case."""

    review: Review
    score: float
    verdict: Verdict


class CaseResult(BaseModel):
    """The case's result; ``review``, ``score`` and ``verdict`` are the first
    reviewer's, and ``ratings`` holds every reviewer's by name."""

    case_id: str
    category: str
    title: str
    world: str
    run: Run
    checks: list[CheckResult]
    review: Review
    score: float
    verdict: Verdict
    ratings: dict[str, Rating] = Field(default_factory=dict)
