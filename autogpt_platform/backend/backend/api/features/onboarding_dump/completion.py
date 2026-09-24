"""Persist and report a completed onboarding brain dump."""

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import OnboardingBrainDump

from backend.api.features.onboarding_dump import db
from backend.util import posthog_client
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

TRANSCRIBED_EVENT = "brain_dump_transcribed"
EXPORT_FAILED_EVENT = "brain_dump_transcript_export_failed"
# The analytics SDK drops events above 900 KiB. Reserve room for its envelope,
# metadata and context; measure escaped JSON, which can exceed UTF-8 text size.
MAX_TRANSCRIPT_JSON_BYTES = 800 * 1024


async def complete_dump(user_id: str, recording_id: str) -> None:
    """Complete one take, then best-effort enqueue its full transcript once.

    Snapshot before the conditional write: a new recording can replace the
    user's row immediately afterwards. A failed analytics read must not hold
    onboarding open. The claim is not a durable delivery outbox.
    """
    try:
        dump = await db.get_dump(user_id)
    except Exception:
        logger.warning("Could not read brain dump analytics for user %s", user_id)
        dump = None

    completed_at = datetime.now(timezone.utc)
    completed = await db.claim_transition(
        user_id,
        recording_id,
        expected=BrainDumpStatus.extracting,
        new=BrainDumpStatus.completed,
        updatedAt=completed_at,
    )
    if (
        not completed
        or dump is None
        or dump.userId != user_id
        or dump.recordingId != recording_id
    ):
        return
    _track_transcript(dump, completed_at)


def _track_transcript(dump: OnboardingBrainDump, completed_at: datetime) -> None:
    """Use server tracking semantics without copying raw text to Sentry/logs."""
    try:
        if dump.inputMode not in (BrainDumpInputMode.voice, BrainDumpInputMode.typed):
            return
        if not dump.transcript or not dump.transcript.strip():
            return
        client = posthog_client.get_posthog_client()
        if client is None:
            return
        properties = _transcript_properties(dump, completed_at)
        event = TRANSCRIBED_EVENT
        transcript_bytes = len(json.dumps(dump.transcript).encode("utf-8"))
        if transcript_bytes > MAX_TRANSCRIPT_JSON_BYTES:
            event = EXPORT_FAILED_EVENT
            properties.update(
                error_code="transcript_too_large",
                transcript_json_bytes=transcript_bytes,
                limit_bytes=MAX_TRANSCRIPT_JSON_BYTES,
            )
            logger.warning(
                "Brain dump transcript exceeds analytics limit for user %s", dump.userId
            )
        else:
            properties["transcript"] = dump.transcript
        event_uuid = str(
            uuid5(NAMESPACE_URL, f"{dump.userId}:{event}:{dump.recordingId}")
        )
        properties["$insert_id"] = event_uuid
        client.capture(
            event=event,
            distinct_id=dump.userId,
            properties=properties,
            timestamp=completed_at,
            uuid=event_uuid,
        )
    except Exception:
        logger.warning("Failed to track brain dump transcript for user %s", dump.userId)


def _transcript_properties(
    dump: OnboardingBrainDump, completed_at: datetime
) -> dict[str, Any]:
    return {
        "environment": Settings().config.app_env.value,
        "source": "platform",
        "user_id": dump.userId,
        "brain_dump_id": dump.id,
        "recording_id": dump.recordingId,
        "input_mode": dump.inputMode.value,
        "duration_seconds": dump.durationSecs,
        "transcript_language": dump.transcriptLang,
        "completed_at": completed_at.isoformat(),
        "transcript_chars": len(dump.transcript or ""),
        "transcript_truncated": False,
    }
