"""Data shapes for the expert style gate: fixtures, rubric, gate, results."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

PromptKind = Literal[
    "briefing", "briefing_lede", "reply_draft", "escalation", "failure"
]
PROMPT_KINDS: tuple[PromptKind, ...] = (
    "briefing",
    "briefing_lede",
    "reply_draft",
    "escalation",
    "failure",
)
PROMPTS_PER_EXPERT = 30

# Which arm produced a response: the expert's own suffix judged against its
# own spec (the number the gate reads), the same response judged against a
# different expert's spec, or plain AutoPilot with no suffix at all. The two
# controls exist to prove the judge separates before the score is trusted.
Arm = Literal["expert", "wrong_spec", "no_suffix"]


class LedeRun(BaseModel):
    agent_name: str
    status: Literal["COMPLETED", "FAILED"]
    title: str


class LedeFacts(BaseModel):
    """Inputs to the morning-briefing narrative, in the shape ``narrative.py``
    reads them from ``BriefingContent``."""

    completed_total: int = 0
    failed_total: int = 0
    decision_total: int = 0
    runs: list[LedeRun] = Field(default_factory=list)


class ReferencePrompt(BaseModel):
    id: str
    kind: PromptKind
    # The user message for chat kinds; ``facts`` for ``briefing_lede``.
    prompt: str = ""
    facts: LedeFacts | None = None

    @model_validator(mode="after")
    def _one_input(self) -> "ReferencePrompt":
        if self.kind == "briefing_lede":
            if self.facts is None or self.prompt:
                raise ValueError(f"{self.id}: briefing_lede takes facts, not prompt")
        elif not self.prompt.strip() or self.facts is not None:
            raise ValueError(f"{self.id}: {self.kind} takes a prompt, not facts")
        return self


class WorkflowFixture(BaseModel):
    name: str
    description: str


class ExpertFixture(BaseModel):
    expert: str
    # The roster preloads, as the hire installs them: they render into the
    # first-turn <expert_workflows> block and answer a library search.
    workflows: list[WorkflowFixture] = Field(default_factory=list)
    prompts: list[ReferencePrompt]


class RubricDimension(BaseModel):
    key: str
    name: str
    question: str
    anchors: dict[str, str]


class Rubric(BaseModel):
    version: int
    scale_min: int
    scale_max: int
    dimensions: list[RubricDimension]


class GateConfig(BaseModel):
    pass_threshold: float
    judge_model: str
    # sha256 of the assembled prompts, resolved models, fixtures and rubric at
    # the last gated run; the paid legs are skipped while it still matches.
    last_gated_fingerprint: str | None = None
    last_gated_at: str | None = None


class DimensionJudgement(BaseModel):
    score: int
    evidence: str = ""


class Judgement(BaseModel):
    scores: dict[str, DimensionJudgement]
    note: str = ""


class Usage(BaseModel):
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float | None = None


class ScoredResponse(BaseModel):
    expert: str
    spec_expert: str
    arm: Arm
    kind: PromptKind
    prompt_id: str
    repeat: int
    model: str
    response: str
    truncated: bool = False
    tool_calls: list[str] = []
    rounds: int = 0
    hit_round_cap: bool = False
    judgement: Judgement | None = None
    score: float | None = None
    error: str | None = None
    generation: Usage | None = None
    judging: Usage | None = None


class Distribution(BaseModel):
    n: int
    mean: float
    sd: float
    min: float
    p25: float
    median: float
    p75: float
    max: float


class ExpertSummary(BaseModel):
    expert: str
    scores: Distribution
    by_kind: dict[str, Distribution]
    by_repeat: list[float]
    below_60: int
    errors: int
    tool_calls: int = 0
    round_cap_hits: int = 0


class PairedAdvantage(BaseModel):
    """Own-spec minus wrong-spec on the SAME response. Pairing removes the
    between-response variance that a raw mean-vs-SD comparison mistakes for
    noise: one turn that never finished drags both arms equally."""

    n: int
    mean: float
    sem: float
    wins: int
    ties: int
    win_rate: float


class Separation(BaseModel):
    right_spec_mean: float
    right_spec_sd: float
    wrong_spec_mean: float | None
    no_suffix_mean: float | None
    gap: float | None
    paired: PairedAdvantage | None = None
    separated: bool | None


class GateOutcome(BaseModel):
    threshold: float
    passed: bool
    failing: list[str]
    skipped: bool = False
    reason: str = ""


class StyleEvalResult(BaseModel):
    run_id: str
    ts: str
    fingerprint: str
    chat_model: str
    lede_model: str
    judge_model: str
    experts: list[ExpertSummary]
    separation: Separation | None
    gate: GateOutcome | None
    cost_usd: float
    cost_known: bool
    input_tokens: int
    output_tokens: int
    responses: list[ScoredResponse]
