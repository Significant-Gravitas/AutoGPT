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
    goal: str | None = None
    done_when: str | None = None
    parent: str | None = None


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


class WorkerBrief(_Open):
    """A turn in a task's worker chat: the brief, wrapped in production's
    worker instructions (``start_task.worker_prompt_text``)."""

    title: str
    brief: str
    task_id: str = "task_new_1"
    done_when: str | None = None


class TaskResult(_Open):
    """A turn that wakes Otto when a task's worker ends, in production's own
    words (``background_delivery.continuation_text``). With a ``summary`` the
    task finished and left it; without one Otto reads ``worker_reply``."""

    title: str = ""
    outcome: str = "completed"
    summary: str | None = None
    document: str | None = None
    worker_reply: str = ""


class Turn(_Open):
    """What starts a turn: the user's message, a worker's brief, or a task's
    result coming back."""

    user: str = ""
    worker: WorkerBrief | None = None
    task_result: TaskResult | None = None


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
    starts_work: bool | None = Field(
        default=None,
        description="Hands the work to the background: start_task, delegating "
        "to an expert, or resuming the user's existing matching task.",
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
    task_has_goal: bool | None = Field(
        default=None,
        description="True: every task started says what it is for (goal) and "
        "what done looks like (done_when).",
    )
    max_started_tasks: int | None = Field(
        default=None,
        description="At most this many new tasks are opened; asking for work "
        "already running, or retrying a task, opens none.",
    )
    retries_task: bool | None = Field(
        default=None,
        description="True: an existing task is run again (start_task with its "
        "task_id) rather than a new one opened.",
    )
    reports_progress: bool | None = Field(
        default=None,
        description="True: the worker says where the task is (report_progress).",
    )
    adds_evidence: bool | None = Field(
        default=None,
        description="True: proof is attached naming the tool run or file "
        "written; False: no proof is attached at all.",
    )
    tools_called_any: list[str] = Field(
        default_factory=list, description="At least one of these tools is called."
    )
    max_reply_chars: int | None = Field(
        default=None,
        description="Everything Otto says in a turn fits in this many "
        "characters, for every turn: the user skims.",
    )
    max_reply_lines: int | None = Field(
        default=None, description="The last turn's reply has at most this many lines."
    )
    max_final_reply_chars: int | None = Field(
        default=None,
        description="A worker's closing message fits in this many characters: "
        "the result is on the task, not in the reply.",
    )
    finishes_task: bool | None = Field(
        default=None,
        description="True: the worker leaves its result with finish_task, "
        "and the platform accepts it.",
    )
    saves_document: bool | None = Field(
        default=None,
        description="True: the full result is saved as a file and named as "
        "finish_task's file.",
    )
    evidence_verified: bool | None = Field(
        default=None,
        description="True: proof is attached and every piece is verified "
        "against what the worker really did.",
    )
    worker_lines_fit: bool | None = Field(
        default=None,
        description="True: no progress line, claim or summary is refused for "
        "its length or its argument names; each lands first time.",
    )
    task_lines_fit: bool | None = Field(
        default=None,
        description="True: every started task's goal and done-when fit their "
        "caps as written, with nothing cut.",
    )
    links_document: bool | None = Field(
        default=None,
        description="True: the last reply links the task's document "
        "(workspace://); False: it links none.",
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
    failed: bool = Field(
        default=False,
        description="The platform answered with an error: a line over its "
        "cap, a file that was never saved, a missing argument.",
    )
    result_preview: str = ""


class Question(BaseModel):
    question: str
    reason: str | None = None
    default_guess: str | None = None


class TurnRecord(BaseModel):
    user: str
    reply: str = ""
    final_reply: str = Field(
        default="", description="The turn's closing message, after its last call."
    )
    tool_calls: list[ToolCall] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    ended_by: Literal[
        "reply", "question", "task_started", "finished", "round_cap", "error"
    ] = "reply"
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
