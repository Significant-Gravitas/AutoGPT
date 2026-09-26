"""One cost row per background LLM call, whatever the job.

Every background call is charged and logged the same way: one
``persist_and_record_usage`` call under its route's provider label and
credential, attributed to the scope's expert, with the job's correlation id
where the admin cost view reads it. A call the provider did not price is
priced here from the catalog price card (``copilot/price_card.py``), at the
route's discount; a model with no catalog price keeps an unknown cost and
logs its tokens without a charge.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from backend.copilot.dream.routing import batch_discount
from backend.copilot.model import ChatSession
from backend.copilot.price_card import compute_cost_usd, price_for
from backend.copilot.token_tracking import persist_and_record_usage

from .context import (
    InferenceContext,
    InferenceJob,
    InferenceKind,
    InferenceUsage,
    Payer,
    RouteDecision,
)


class _KindAccounting(BaseModel):
    """How one kind of job appears in the cost log."""

    model_config = ConfigDict(frozen=True)

    # The row's metadata ``source``, which the admin view filters on.
    source: str
    # Whether the spend counts against the user's interactive daily budget,
    # or only the weekly one, as background work the user never asked for
    # turn by turn does.
    charges_daily: bool
    # Where the job's correlation id goes on the row.
    correlation_column: Literal["graph_exec", "chat_session", "none"]
    # The row's ``execution_path`` for a sync call. The dream's has always
    # read ``sync_baseline`` (the admin view's "(dream)" path filter); every
    # other job's reads ``sync``, like a chat turn's.
    sync_path_label: Literal["sync_baseline", "sync"] = "sync"


_ACCOUNTING: dict[InferenceKind, _KindAccounting] = {
    # A pass's phase rows join under its id; the admin view reads a
    # ``graphExecId`` as a run only on ``source="dream_pass"`` rows.
    "dream": _KindAccounting(
        source="dream_pass",
        charges_daily=False,
        correlation_column="graph_exec",
        sync_path_label="sync_baseline",
    ),
    # On any other source the admin view reads a ``graphExecId`` as a chat
    # logged before ``chatSessionId`` existed, so the briefing's correlation
    # stays in its trace.
    "briefing_narrative": _KindAccounting(
        source="morning_briefing", charges_daily=False, correlation_column="none"
    ),
    # A consult is part of the chat turn that asked for it.
    "consult": _KindAccounting(
        source="copilot", charges_daily=True, correlation_column="chat_session"
    ),
    "eval_judge": _KindAccounting(
        source="expert_style_eval", charges_daily=False, correlation_column="none"
    ),
}

_BILLING_MODES: dict[Payer, str] = {
    "platform_allowance": "platform",
    "local": "local",
}


async def record(
    ctx: InferenceContext,
    usage: InferenceUsage,
    *,
    block_name: str,
    metadata: dict[str, Any] | None = None,
    session: ChatSession | None = None,
) -> InferenceUsage:
    """Charge *usage* to the scope's user and log its cost row.

    *metadata* adds the caller's own keys; the record's keys win over them.
    *session* is the chat a consult ran in, whose usage list the tokens join.
    Returns *usage* as priced. No row and no charge when the call used no
    tokens and has no cost.
    """
    priced = price(usage, ctx.route)
    accounting = _ACCOUNTING[ctx.job.kind]
    graph_exec_id, chat_session_id = _correlation_columns(ctx.job, accounting)
    await persist_and_record_usage(
        session=session,
        user_id=ctx.scope.user_id,
        prompt_tokens=priced.input_tokens,
        completion_tokens=priced.output_tokens,
        cache_read_tokens=priced.cache_read_tokens,
        cache_creation_tokens=priced.cache_creation_tokens,
        log_prefix=f"[{ctx.job.label}]",
        cost_usd=priced.cost_usd,
        model=priced.model,
        provider=ctx.route.cost_log_provider,
        block_name_override=block_name,
        extra_metadata={**(metadata or {}), **_row_metadata(ctx, accounting)},
        graph_exec_id_override=graph_exec_id,
        chat_session_id_override=chat_session_id,
        credential_id_override=ctx.route.credential_id,
        skip_daily=not accounting.charges_daily,
        expert_id=ctx.scope.expert_id,
    )
    return priced


def price(usage: InferenceUsage, route: RouteDecision) -> InferenceUsage:
    """*usage* with its cost read off the catalog price card when the
    provider reported none: each token bucket at the model's list rate, less
    the route's discount. A local backend's calls stay unpriced; they bill
    nobody."""
    if usage.cost_usd is not None or route.payer == "local":
        return usage
    card = price_for(usage.model)
    if card is None:
        return usage
    cost = compute_cost_usd(
        price=card,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_creation_tokens=usage.cache_creation_tokens,
        discount=batch_discount(route.execution_path),
    )
    return usage.model_copy(update={"cost_usd": cost, "cost_source": "catalog"})


def _correlation_columns(
    job: InferenceJob, accounting: _KindAccounting
) -> tuple[str | None, str | None]:
    """``(graph_exec_id, chat_session_id)`` for the job's row."""
    match accounting.correlation_column:
        case "graph_exec":
            return job.correlation_id, None
        case "chat_session":
            return None, job.correlation_id
        case "none":
            return None, None


def _row_metadata(ctx: InferenceContext, accounting: _KindAccounting) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": accounting.source,
        "job_kind": ctx.job.kind,
        "phase": ctx.job.phase,
        "execution_path": _logged_path(ctx, accounting),
        "billing_mode": _BILLING_MODES[ctx.route.payer],
        "discount_applied": batch_discount(ctx.route.execution_path),
        "expert_id": ctx.scope.expert_id,
    }
    if ctx.trace_id:
        metadata["langfuse_trace_id"] = ctx.trace_id
    return metadata


def _logged_path(ctx: InferenceContext, accounting: _KindAccounting) -> str:
    if ctx.route.execution_path == "sync_baseline":
        return accounting.sync_path_label
    return ctx.route.execution_path
