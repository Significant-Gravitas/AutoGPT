"""Aggregate historical block-execution credit costs from CreditTransaction.

Used by the admin export endpoint to seed `block_preflight_estimates.json`,
giving dynamic-cost blocks a non-zero pre-flight charge so post-flight
reconciliation only settles a small delta.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel

from backend.blocks import get_block
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.db import query_raw_with_schema

# Match the cost types whose pre-flight is genuinely 0 today (see
# `block_usage_cost` in executor/utils.py). TOKENS is excluded because
# `compute_token_credits` already provides a per-model floor at pre-flight,
# which is more accurate than a per-block historical average that mixes
# Haiku-class and Opus-class executions.
DYNAMIC_COST_TYPES = {"second", "items", "cost_usd"}

ANALYTICS_MAX_DAYS = 90
ANALYTICS_MIN_SAMPLES_DEFAULT = 10


class BlockCostEstimateRow(BaseModel):
    block_id: str
    block_name: str
    cost_type: str
    samples: int
    mean_credits: int
    p50_credits: int
    p95_credits: int


def _resolve_cost_type(block_id: str) -> str | None:
    """Return the first dynamic cost type registered for this block, if any.

    A block can declare multiple BlockCost entries with different cost_filters.
    We collapse them to a single representative type for the export so the
    admin view stays compact; reconciliation continues to use the per-filter
    entry at runtime.

    Trade-off: the JSON estimate is **per-block-id, not per-filter**. For
    blocks with tier-style filters (e.g. Ayrshare `is_video=True → 5cr` vs
    `is_video=False → 2cr`), the seeded mean averages across all filters and
    the pre-flight charge will over-bill low-tier runs and under-bill high-tier
    runs; reconciliation refunds/debt settles the delta. The seeding-PR
    reviewer should spot-check the `p95/mean` ratio for any block whose ratio
    is wide enough to suggest filter-tier costs.
    """
    block = get_block(block_id)
    if not block:
        return None
    costs = BLOCK_COSTS.get(type(block), [])
    for c in costs:
        if c.cost_type.value in DYNAMIC_COST_TYPES:
            return c.cost_type.value
    return None


async def compute_block_cost_estimates(
    *,
    start: datetime,
    end: datetime,
    min_samples: int = ANALYTICS_MIN_SAMPLES_DEFAULT,
) -> list[BlockCostEstimateRow]:
    """Aggregate per-(block_id, node_exec_id) credit cost over [start, end].

    Sums all USAGE rows per node-execution (capturing pre-flight + reconciliation
    deltas as one cost), then averages across executions per block. Filters to
    blocks whose current cost type is dynamic — static-cost blocks already
    charge correctly pre-flight and don't need an estimate.

    User-ID exemption: this query intentionally aggregates across users — the
    purpose is platform-wide cost calibration, not per-user data access. Per
    AGENTS.md the caller must enforce admin-only auth (the only caller is the
    admin route, which is gated by `requires_admin_user`).
    """
    if start >= end:
        raise ValueError("start must be before end")
    # Use total_seconds rather than `.days` so a 90d-and-1-hour window doesn't
    # slip past the cap because `.days` truncates fractional days.
    if (end - start) > timedelta(days=ANALYTICS_MAX_DAYS):
        approx_days = (end - start).total_seconds() / 86400
        raise ValueError(f"window {approx_days:.2f}d exceeds max {ANALYTICS_MAX_DAYS}d")

    query = """
    WITH per_exec AS (
      SELECT
        metadata->>'block_id' AS block_id,
        MAX(metadata->>'block') AS block_name,
        metadata->>'node_exec_id' AS node_exec_id,
        SUM(-amount) AS exec_cost
      FROM {schema_prefix}"CreditTransaction"
      WHERE type = 'USAGE'
        AND "createdAt" >= $1::timestamp
        AND "createdAt" <= $2::timestamp
        AND "isActive" = true
        AND metadata->>'block_id' IS NOT NULL
        AND metadata->>'node_exec_id' IS NOT NULL
      GROUP BY metadata->>'block_id', metadata->>'node_exec_id'
    )
    SELECT
      block_id,
      MAX(block_name) AS block_name,
      COUNT(*)::int AS samples,
      ROUND(AVG(exec_cost))::int AS mean_credits,
      ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY exec_cost))::int AS p50_credits,
      ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY exec_cost))::int AS p95_credits
    FROM per_exec
    -- Keep zero-cost executions (post-flight refund cancelled the pre-flight
    -- estimate) so the historical mean reflects the full distribution. Drop
    -- only over-refunded outliers, which are net-negative on the user side.
    WHERE exec_cost >= 0
    GROUP BY block_id
    HAVING COUNT(*) >= $3
    ORDER BY samples DESC
    """
    rows: list[dict[str, Any]] = await query_raw_with_schema(
        query, start, end, min_samples
    )

    out: list[BlockCostEstimateRow] = []
    for r in rows:
        block_id = r["block_id"]
        cost_type = _resolve_cost_type(block_id)
        if not cost_type:
            # Skip blocks that don't have a dynamic cost type — their pre-flight
            # is already correct and shouldn't be overridden by a historical mean.
            continue
        # The SQL guards against NULL aggregates (HAVING COUNT >= 1, exec_cost >= 0)
        # but a single corrupt row could still surface as NULL — coerce defensively
        # so a malformed row doesn't 500 the endpoint.
        out.append(
            BlockCostEstimateRow(
                block_id=block_id,
                block_name=r.get("block_name") or block_id,
                cost_type=cost_type,
                samples=int(r.get("samples") or 0),
                mean_credits=int(r.get("mean_credits") or 0),
                p50_credits=int(r.get("p50_credits") or 0),
                p95_credits=int(r.get("p95_credits") or 0),
            )
        )
    return out
