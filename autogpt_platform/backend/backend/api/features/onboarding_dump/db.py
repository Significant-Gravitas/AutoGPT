"""Persistence for the onboarding brain dump.

One row per user (``userId`` is unique): re-recording replaces the take in
place, and ``recordingId`` identifies the current one so ``finalize`` can
be retried safely.
"""

from datetime import datetime
from typing import Any

from prisma import Json
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import OnboardingBrainDump


async def get_dump(user_id: str) -> OnboardingBrainDump | None:
    return await OnboardingBrainDump.prisma().find_unique(where={"userId": user_id})


async def owns_dump(user_id: str, recording_id: str) -> bool:
    """Whether ``recording_id`` is still the take the row belongs to.

    Row writes carry their own ``recordingId`` guard, but the business
    understanding is shared user context with no take on it. A background
    job for a superseded take has to be told to stop before it writes
    there, and this is what tells it.
    """
    dump = await get_dump(user_id)
    return dump is not None and dump.recordingId == recording_id


# A take still being processed. A new typed or skipped take cannot displace
# one: unlike part 0 of a voice upload, they carry no earlier server-side
# event that can establish which tab's request is newer.
_PROCESSING_STATUSES = frozenset(
    {
        BrainDumpStatus.recording_uploaded,
        BrainDumpStatus.transcribing,
        BrainDumpStatus.transcribed,
        BrainDumpStatus.extracting,
    }
)

# Re-claiming the same take in any of these would knock it back to the start
# of the pipeline. ``failed`` is deliberately absent: retrying it must reset it.
_IN_FLIGHT_STATUSES = _PROCESSING_STATUSES | {BrainDumpStatus.completed}

# Everything a single take owns. One row per user means a new take
# inherits whatever the last one left here, so unless these are cleared
# ``/recording``, ``/intro``, ``/recommended-providers`` and
# ``/recommended-experts`` keep serving the *previous* take's audio,
# transcript, greeting, picks and team whenever the new one fails, is
# skipped or is abandoned half-recorded.
#
# ``greetingSeen`` belongs with them. It is only ever written True and
# short-circuits the intro endpoint, so a new take that inherited it would
# run the whole pipeline for a greeting that could never render. Clearing
# it is only safe *because* the transcript goes too: a take that never
# produces a greeting leaves nothing to reflect back, so the intro falls
# through to Path B rather than waiting on a greeting that isn't coming.
_TAKE_OWNED_RESET: dict[str, Any] = {
    "audioPath": None,
    "mimeType": None,
    "sizeBytes": None,
    "durationSecs": None,
    "transcript": None,
    "transcriptLang": None,
    "greeting": None,
    "suggestedPrompts": Json([]),
    "recommendedProviders": Json(None),
    "recommendedExperts": Json(None),
    "greetingSeen": False,
}


async def start_dump(
    user_id: str,
    recording_id: str,
    input_mode: BrainDumpInputMode,
) -> OnboardingBrainDump:
    """Claim the row for ``recording_id``, resetting any prior take's state.

    A take already moving through the pipeline is returned untouched when
    recovery replays part 0 or finalize is retried, and a typed or skipped
    take with a different id cannot displace one still being processed. Any
    other new id is a deliberate new take, a re-take after a finished one
    included, and claims the row, taking every take-owned column with it.
    """
    existing = await get_dump(user_id)
    if existing and (
        (
            existing.recordingId == recording_id
            and existing.status in _IN_FLIGHT_STATUSES
        )
        or (
            input_mode != BrainDumpInputMode.voice
            and existing.status in _PROCESSING_STATUSES
        )
    ):
        return existing

    is_new_take = existing is None or existing.recordingId != recording_id
    fields: dict[str, Any] = {
        "recordingId": recording_id,
        "status": BrainDumpStatus.recording_uploaded,
        "inputMode": input_mode,
        "errorCode": None,
        "errorReason": None,
        **(_TAKE_OWNED_RESET if is_new_take else {}),
    }
    return await OnboardingBrainDump.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": {"userId": user_id, **_TAKE_OWNED_RESET, **fields},
            "update": fields,
        },
    )


async def update_dump(user_id: str, recording_id: str, **fields: Any) -> bool:
    """Advance one take's row, and only that take's.

    ``recordingId`` is part of the ``WHERE`` on purpose. There is one row
    per user, so a second tab starting a new take moves the row on while
    the old take's transcription is still running — minutes, for a long
    dump. Scoped on ``userId`` alone, that old take's transcript, greeting
    and failure code all land on the *new* take's row, and the new
    recording is never transcribed at all.

    Returns ``False`` when nothing matched: no row (a status update for a
    take that was never started is a no-op, not an error, so a
    late-arriving retry can't resurrect a deleted user's dump) or a take
    that has since been superseded.
    """
    updated = await OnboardingBrainDump.prisma().update_many(
        where={"userId": user_id, "recordingId": recording_id},
        data=fields,
    )
    return updated == 1


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


async def mark_failed(
    user_id: str, recording_id: str, error_code: str, reason: str | None = None
) -> None:
    """Record a terminal failure without ever dropping the stored audio.

    ``reason`` says what decided a quality-gate rejection; any other failure
    clears the one an earlier attempt left.
    """
    await update_dump(
        user_id,
        recording_id,
        status=BrainDumpStatus.failed,
        errorCode=error_code,
        errorReason=reason,
    )


async def mark_greeting_seen(user_id: str) -> None:
    """Set the greeting-done flag, creating a stub row when none exists.

    Users who never reached the dump (or whose browser lost the local
    flag) still need "seen" to stick server-side, so this upserts a
    minimal skipped-completed row rather than silently no-opping.
    """
    await OnboardingBrainDump.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": {**_greeting_only_row(user_id), "greetingSeen": True},
            "update": {"greetingSeen": True},
        },
    )


async def ensure_dump(user_id: str) -> OnboardingBrainDump:
    """The user's row, creating the same stub as ``mark_greeting_seen`` if absent."""
    return await OnboardingBrainDump.prisma().upsert(
        where={"userId": user_id},
        data={"create": _greeting_only_row(user_id), "update": {}},
    )


def _greeting_only_row(user_id: str) -> dict[str, Any]:
    return {
        "userId": user_id,
        **_TAKE_OWNED_RESET,
        "recordingId": "greeting-only",
        "status": BrainDumpStatus.completed,
        "inputMode": BrainDumpInputMode.skipped,
    }


async def replace_suggested_prompts(
    user_id: str, read_at: datetime, prompts: list[dict[str, Any]]
) -> bool:
    """Write the prompt list, unless the row changed since it was read at ``read_at``.

    Returns ``False`` when another write landed first, so the caller can
    re-read rather than overwrite a dismiss or a send from another tab.
    """
    updated = await OnboardingBrainDump.prisma().update_many(
        where={"userId": user_id, "updatedAt": read_at},
        data={"suggestedPrompts": Json(prompts)},
    )
    return updated == 1


# Postgres caps a statement at 32,767 bind parameters, and the first sweep
# after this shipped clears every recording ever stored.
_CLEAR_AUDIO_BATCH = 1000


async def clear_audio_paths(audio_paths: list[str]) -> int:
    """Forget recordings that are about to be deleted from storage.

    Only ``audioPath`` goes: the transcript is kept while the account
    exists, and the size and duration say nothing about what was said.
    Returns how many rows pointed at one of ``audio_paths``.
    """
    batches = [
        audio_paths[start : start + _CLEAR_AUDIO_BATCH]
        for start in range(0, len(audio_paths), _CLEAR_AUDIO_BATCH)
    ]
    cleared = 0
    for batch in batches:
        cleared += await OnboardingBrainDump.prisma().update_many(
            where={"audioPath": {"in": batch}},
            data={"audioPath": None},
        )
    return cleared
