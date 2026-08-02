"""Persistence for the onboarding brain dump.

One row per user (``userId`` is unique): re-recording replaces the take in
place, and ``recordingId`` identifies the current one so ``finalize`` can
be retried safely.
"""

from typing import Any

from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import OnboardingBrainDump


async def get_dump(user_id: str) -> OnboardingBrainDump | None:
    return await OnboardingBrainDump.prisma().find_unique(where={"userId": user_id})


# A take that has reached any of these is being processed, or is already
# done. Re-claiming it would knock it back to the start of the pipeline.
# ``failed`` is deliberately absent: retrying a failed take must reset it.
_IN_FLIGHT_STATUSES = frozenset(
    {
        BrainDumpStatus.transcribing,
        BrainDumpStatus.transcribed,
        BrainDumpStatus.extracting,
        BrainDumpStatus.completed,
    }
)


async def start_dump(
    user_id: str,
    recording_id: str,
    input_mode: BrainDumpInputMode,
) -> OnboardingBrainDump:
    """Claim the row for ``recording_id``, resetting any prior take's state.

    A take already moving through the pipeline is returned untouched. Two
    callers reach here after a dump has been finalized — recovery replays
    every part on disk, part 0 included, and a repeated finalize — and
    neither should reset an in-flight transcription to
    ``recording_uploaded``. A *different* recording id is a genuinely new
    take and always claims the row.
    """
    existing = await get_dump(user_id)
    if (
        existing
        and existing.recordingId == recording_id
        and existing.status in _IN_FLIGHT_STATUSES
    ):
        return existing

    fields: dict[str, Any] = {
        "recordingId": recording_id,
        "status": BrainDumpStatus.recording_uploaded,
        "inputMode": input_mode,
        "errorCode": None,
    }
    return await OnboardingBrainDump.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": {"userId": user_id, **fields},
            "update": fields,
        },
    )


async def update_dump(user_id: str, **fields: Any) -> OnboardingBrainDump | None:
    """Advance the user's dump row.

    Returns ``None`` when there is no row — a status update for a take
    that was never started is a no-op, not an error, so a late-arriving
    retry can't resurrect a deleted user's dump.
    """
    return await OnboardingBrainDump.prisma().update(
        where={"userId": user_id},
        data=fields,
    )


async def claim_transition(
    user_id: str,
    recording_id: str,
    *,
    expected: BrainDumpStatus,
    new: BrainDumpStatus,
    **fields: Any,
) -> bool:
    """Move a take from ``expected`` to ``new``, once.

    Returns ``True`` only for the caller that won. The read-then-write
    idempotency guards in the service can both be passed by two finalize
    requests that arrive together; this cannot, because the status is
    part of the ``WHERE`` clause, so the database decides the winner.
    """
    updated = await OnboardingBrainDump.prisma().update_many(
        where={
            "userId": user_id,
            "recordingId": recording_id,
            "status": expected,
        },
        data={"status": new, **fields},
    )
    return updated == 1


async def mark_failed(user_id: str, error_code: str) -> None:
    """Record a terminal failure without ever dropping the stored audio."""
    await update_dump(user_id, status=BrainDumpStatus.failed, errorCode=error_code)


async def mark_greeting_seen(user_id: str) -> None:
    """Set the greeting-done flag, creating a stub row when none exists.

    Users who never reached the dump (or whose browser lost the local
    flag) still need "seen" to stick server-side, so this upserts a
    minimal skipped-completed row rather than silently no-opping.
    """
    await OnboardingBrainDump.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": {
                "userId": user_id,
                "recordingId": "greeting-only",
                "status": BrainDumpStatus.completed,
                "inputMode": BrainDumpInputMode.skipped,
                "greetingSeen": True,
            },
            "update": {"greetingSeen": True},
        },
    )
