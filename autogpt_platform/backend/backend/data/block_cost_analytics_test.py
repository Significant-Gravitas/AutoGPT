from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from backend.data.block_cost_analytics import (
    ANALYTICS_MAX_DAYS,
    compute_block_cost_estimates,
)


@pytest.mark.asyncio
async def test_window_cap_rejected():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=ANALYTICS_MAX_DAYS + 1)
    with pytest.raises(ValueError, match="exceeds max"):
        await compute_block_cost_estimates(start=start, end=end)


@pytest.mark.asyncio
async def test_inverted_window_rejected():
    end = datetime.now(timezone.utc)
    start = end + timedelta(hours=1)
    with pytest.raises(ValueError, match="start must be before end"):
        await compute_block_cost_estimates(start=start, end=end)


@pytest.mark.asyncio
async def test_static_cost_blocks_filtered_out():
    """A row whose block resolves to a RUN/BYTE cost block must not appear in
    the response — those types already have a valid pre-flight."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)

    raw_rows = [
        {
            "block_id": "static-run-block",
            "block_name": "RunOnlyBlock",
            "samples": 50,
            "mean_credits": 3,
            "p50_credits": 3,
            "p95_credits": 3,
        },
    ]

    with (
        patch(
            "backend.data.block_cost_analytics.query_raw_with_schema",
            new_callable=AsyncMock,
            return_value=raw_rows,
        ),
        patch(
            "backend.data.block_cost_analytics._resolve_cost_type",
            return_value=None,
        ),
    ):
        out = await compute_block_cost_estimates(start=start, end=end)

    assert out == []


@pytest.mark.asyncio
async def test_dynamic_cost_blocks_returned_with_resolved_type():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)

    raw_rows = [
        {
            "block_id": "dyn-1",
            "block_name": "DynamicSearchBlock",
            "samples": 200,
            "mean_credits": 7,
            "p50_credits": 6,
            "p95_credits": 12,
        },
        {
            "block_id": "dyn-2",
            "block_name": "DynamicCostBlock",
            "samples": 30,
            "mean_credits": 50,
            "p50_credits": 40,
            "p95_credits": 80,
        },
    ]

    type_lookup = {"dyn-1": "second", "dyn-2": "cost_usd"}

    with (
        patch(
            "backend.data.block_cost_analytics.query_raw_with_schema",
            new_callable=AsyncMock,
            return_value=raw_rows,
        ),
        patch(
            "backend.data.block_cost_analytics._resolve_cost_type",
            side_effect=lambda bid: type_lookup.get(bid),
        ),
    ):
        out = await compute_block_cost_estimates(start=start, end=end)

    assert len(out) == 2
    assert out[0].block_id == "dyn-1"
    assert out[0].cost_type == "second"
    assert out[0].mean_credits == 7
    assert out[1].block_id == "dyn-2"
    assert out[1].cost_type == "cost_usd"
    assert out[1].mean_credits == 50


@pytest.mark.asyncio
async def test_window_cap_catches_fractional_overflow():
    """A window of 90 days + 1 hour must trip the cap — the older `.days`
    check truncated and silently let the over-cap window through."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=ANALYTICS_MAX_DAYS, hours=1)
    with pytest.raises(ValueError, match="exceeds max"):
        await compute_block_cost_estimates(start=start, end=end)


@pytest.mark.asyncio
async def test_min_samples_threshold_passed_to_sql_query():
    """`min_samples` is enforced by `HAVING COUNT(*) >= $3` in the SQL — the
    aggregator must forward the parameter unchanged so rows below the threshold
    never surface from the database."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)

    with (
        patch(
            "backend.data.block_cost_analytics.query_raw_with_schema",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_query,
        patch(
            "backend.data.block_cost_analytics._resolve_cost_type",
            return_value="second",
        ),
    ):
        await compute_block_cost_estimates(start=start, end=end, min_samples=42)

    # query_raw_with_schema(query, start, end, min_samples)
    args = mock_query.await_args.args
    assert args[3] == 42


@pytest.mark.asyncio
async def test_block_name_falls_back_to_block_id_when_metadata_missing():
    """Old USAGE rows may not have populated metadata->>'block', leading the
    raw SQL to return a NULL block_name. Verify we fall back to block_id
    rather than emitting `null`."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)

    raw_rows = [
        {
            "block_id": "dyn-1",
            "block_name": None,
            "samples": 12,
            "mean_credits": 4,
            "p50_credits": 4,
            "p95_credits": 5,
        }
    ]

    with (
        patch(
            "backend.data.block_cost_analytics.query_raw_with_schema",
            new_callable=AsyncMock,
            return_value=raw_rows,
        ),
        patch(
            "backend.data.block_cost_analytics._resolve_cost_type",
            return_value="items",
        ),
    ):
        out = await compute_block_cost_estimates(start=start, end=end)

    assert len(out) == 1
    # Falls back to block_id when block_name is null in the historical row
    # (not the block class name — the resolver isn't called here because the
    # row already supplied a block_id; we just want a non-null label).
    assert out[0].block_name == "dyn-1"
