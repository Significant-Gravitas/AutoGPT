"""What one background LLM call is: who it runs for, the job it does, the
route it takes and what it used.

A chat turn carries all of this in its session. The background calls this
package serves (a dream phase, the briefing's lede, a consult, an eval judge;
see ``__init__.py`` for the ones not moved yet) have no session, so each
builds an ``InferenceContext`` up front and hands it to
``complete.structured_complete``, ``trace.trace`` and ``record.record``, which
read attribution, routing and accounting off it instead of each caller
working them out again.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.copilot.dream.routing import ExecutionPath
from backend.util.llm.providers import ProviderLiteral

InferenceKind = Literal["dream", "briefing_narrative", "consult", "eval_judge"]
LatencyClass = Literal["deferred", "bounded"]
# The deployment's fast standard or fast advanced model, or ``aux``: the cheap
# auxiliary model (``title_model``) the one-shot jobs have always used.
InferenceTier = Literal["standard", "advanced", "aux"]
# ``provider_batch`` is the dream's Anthropic batch path, which the
# orchestrator still picks itself; its rows are recorded through here too.
Engine = Literal["provider_sync", "provider_batch"]
Payer = Literal["platform_allowance", "local"]
CostSource = Literal["provider", "catalog", "none"]


class InferenceScope(BaseModel):
    """Who a call runs for: the account, and the expert when it is one's."""

    model_config = ConfigDict(frozen=True)

    user_id: str = Field(min_length=1)
    expert_id: str | None = Field(default=None, min_length=1)


class InferenceJob(BaseModel):
    """What a call is for, whatever route it takes.

    ``correlation_id`` joins the calls of one unit of work: a dream pass, a
    chat (for a consult), one briefing, one eval run. ``pinned_model`` is for
    a job that must run on one model whatever its tier: an eval judge's scores
    only compare against a baseline judged by the same model.
    """

    model_config = ConfigDict(frozen=True)

    kind: InferenceKind
    phase: str | None = None
    correlation_id: str = Field(min_length=1)
    latency_class: LatencyClass
    tier: InferenceTier
    timeout_seconds: float | None = None
    pinned_model: str | None = None

    @property
    def label(self) -> str:
        """``kind:phase``, or the kind alone: the name the job's trace and
        log lines go by."""
        return f"{self.kind}:{self.phase}" if self.phase else self.kind


class RouteDecision(BaseModel):
    """How a call is made and who pays for it.

    Only the routes that exist today: the platform's own key on the chat
    transport, and the dream's Anthropic batch path. ``credential_id`` is
    ``None`` for the platform's key; the cost log records that as the copilot
    system credential.
    """

    model_config = ConfigDict(frozen=True)

    engine: Engine
    auth_provider: Literal["platform"]
    credential_id: str | None = None
    provider: ProviderLiteral
    model: str
    payer: Payer
    execution_path: ExecutionPath
    cost_log_provider: str
    reason: str


class InferenceUsage(BaseModel):
    """What one call used, and what it cost when that is known.

    ``cost_source`` says where ``cost_usd`` came from: the provider's own
    figure, the catalog price card, or ``none`` when the cost is unknown
    (never zero).
    """

    model_config = ConfigDict(frozen=True)

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float | None = None
    cost_source: CostSource = "none"
    payer: Payer

    @model_validator(mode="after")
    def _cost_and_source_agree(self) -> "InferenceUsage":
        if (self.cost_usd is None) != (self.cost_source == "none"):
            raise ValueError("cost_source is 'none' exactly when cost_usd is None")
        return self


class InferenceContext(BaseModel):
    """One call's scope, job and route; ``trace_id`` once it is traced."""

    model_config = ConfigDict(frozen=True)

    scope: InferenceScope
    job: InferenceJob
    route: RouteDecision
    trace_id: str | None = None


class InferenceError(RuntimeError):
    """A background call produced no usable answer.

    ``usage`` is the provider's spend when the failure came *after* a response
    (empty content, unparseable JSON, a schema mismatch): those tokens were
    billed, so a caller keeping a cost ledger records them. ``None`` means no
    call completed and nothing was charged.
    """

    def __init__(self, message: str, usage: InferenceUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage
