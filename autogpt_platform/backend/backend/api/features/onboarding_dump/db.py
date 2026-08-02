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


async def start_dump(
    user_id: str,
    recording_id: str,
    input_mode: BrainDumpInputMode,
) -> OnboardingBrainDump:
    """Claim the row for ``recording_id``, resetting any prior take's state."""
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
