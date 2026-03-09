"""Tests for the understanding prompt backfill script."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from scripts.backfill_understanding_prompts import backfill_understanding_prompts


def make_record(*, user_id: str, business: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"understanding-{user_id}",
        userId=user_id,
        createdAt=datetime.now(timezone.utc),
        updatedAt=datetime.now(timezone.utc),
        data={"business": business},
    )


@pytest.mark.asyncio
async def test_backfill_understanding_prompts_dry_run():
    record = make_record(
        user_id="user-1",
        business={"business_name": "Acme", "industry": "Support"},
    )
    prisma = AsyncMock()
    prisma.find_many.side_effect = [[record], []]

    with (
        patch(
            "scripts.backfill_understanding_prompts.CoPilotUnderstanding.prisma",
            return_value=prisma,
        ),
        patch(
            "scripts.backfill_understanding_prompts.generate_understanding_prompts",
            new_callable=AsyncMock,
        ) as mock_generate,
        patch(
            "scripts.backfill_understanding_prompts.update_business_understanding_prompts",
            new_callable=AsyncMock,
        ) as mock_update,
    ):
        summary = await backfill_understanding_prompts(batch_size=10, dry_run=True)

    assert summary == {
        "scanned": 1,
        "candidates": 1,
        "eligible": 1,
        "updated": 0,
        "failed": 0,
        "skipped_existing": 0,
        "skipped_no_context": 0,
    }
    mock_generate.assert_not_awaited()
    mock_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_backfill_understanding_prompts_skips_existing_prompts():
    record = make_record(
        user_id="user-1",
        business={
            "business_name": "Acme",
            "prompts": ["Prompt one", "Prompt two", "Prompt three"],
        },
    )
    prisma = AsyncMock()
    prisma.find_many.side_effect = [[record], []]

    with patch(
        "scripts.backfill_understanding_prompts.CoPilotUnderstanding.prisma",
        return_value=prisma,
    ):
        summary = await backfill_understanding_prompts(batch_size=10)

    assert summary == {
        "scanned": 1,
        "candidates": 0,
        "eligible": 0,
        "updated": 0,
        "failed": 0,
        "skipped_existing": 1,
        "skipped_no_context": 0,
    }


@pytest.mark.asyncio
async def test_backfill_understanding_prompts_updates_missing_prompts():
    record = make_record(
        user_id="user-1",
        business={"business_name": "Acme", "industry": "Support"},
    )
    prisma = AsyncMock()
    prisma.find_many.side_effect = [[record], []]

    with (
        patch(
            "scripts.backfill_understanding_prompts.CoPilotUnderstanding.prisma",
            return_value=prisma,
        ),
        patch(
            "scripts.backfill_understanding_prompts.generate_understanding_prompts",
            new_callable=AsyncMock,
            return_value=["Prompt one", "Prompt two", "Prompt three"],
        ) as mock_generate,
        patch(
            "scripts.backfill_understanding_prompts.update_business_understanding_prompts",
            new_callable=AsyncMock,
            return_value=object(),
        ) as mock_update,
    ):
        summary = await backfill_understanding_prompts(batch_size=10)

    assert summary == {
        "scanned": 1,
        "candidates": 1,
        "eligible": 1,
        "updated": 1,
        "failed": 0,
        "skipped_existing": 0,
        "skipped_no_context": 0,
    }
    mock_generate.assert_awaited_once()
    mock_update.assert_awaited_once_with(
        "user-1", ["Prompt one", "Prompt two", "Prompt three"]
    )


@pytest.mark.asyncio
async def test_backfill_understanding_prompts_skips_records_without_context():
    record = make_record(user_id="user-1", business={})
    prisma = AsyncMock()
    prisma.find_many.side_effect = [[record], []]

    with (
        patch(
            "scripts.backfill_understanding_prompts.CoPilotUnderstanding.prisma",
            return_value=prisma,
        ),
        patch(
            "scripts.backfill_understanding_prompts.generate_understanding_prompts",
            new_callable=AsyncMock,
        ) as mock_generate,
    ):
        summary = await backfill_understanding_prompts(batch_size=10)

    assert summary == {
        "scanned": 1,
        "candidates": 1,
        "eligible": 0,
        "updated": 0,
        "failed": 0,
        "skipped_existing": 0,
        "skipped_no_context": 1,
    }
    mock_generate.assert_not_awaited()
