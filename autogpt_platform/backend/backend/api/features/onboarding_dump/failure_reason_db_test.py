"""A quality rejection's reason against Postgres.

The service and route tests fake ``mark_failed`` and ``start_dump``, so only
the database shows the column is written, and cleared once it stops applying.
"""

import uuid
from collections.abc import AsyncGenerator

import pytest_asyncio
from prisma.enums import BrainDumpInputMode
from prisma.models import User

from backend.api.features.onboarding_dump import db

RECORDING_ID = "rec-1"


@pytest_asyncio.fixture(loop_scope="session")
async def user_id(server) -> AsyncGenerator[str, None]:
    user_id = str(uuid.uuid4())
    await User.prisma().create(data={"id": user_id, "email": f"{user_id}@example.com"})
    yield user_id
    await User.prisma().delete(where={"id": user_id})


async def test_a_rejection_stores_its_reason_until_the_take_is_retried(user_id: str):
    await db.start_dump(user_id, RECORDING_ID, BrainDumpInputMode.voice)
    await db.mark_failed(
        user_id, RECORDING_ID, "quality_check_unavailable", "judge_timeout"
    )

    rejected = await db.get_dump(user_id)
    assert rejected is not None
    assert rejected.errorReason == "judge_timeout"

    await db.start_dump(user_id, RECORDING_ID, BrainDumpInputMode.voice)

    retried = await db.get_dump(user_id)
    assert retried is not None
    assert retried.errorReason is None


async def test_a_failure_outside_the_gate_clears_an_earlier_reason(user_id: str):
    await db.start_dump(user_id, RECORDING_ID, BrainDumpInputMode.voice)
    await db.mark_failed(
        user_id, RECORDING_ID, "insufficient_content", "judge_unusable"
    )
    await db.mark_failed(user_id, RECORDING_ID, "transcription_failed")

    failed = await db.get_dump(user_id)
    assert failed is not None
    assert (failed.errorCode, failed.errorReason) == ("transcription_failed", None)
