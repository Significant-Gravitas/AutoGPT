"""Inbound attachment upload + user/model-facing failure notes.

The adapter already downloaded the platform files (bounded by its caps); this
uploads them into the turn's workspace session and turns per-file failures into
the two notes every platform shares: one telling the user, one telling the
model which files it does NOT have.
"""

import logging

from .adapters.base import MessageContext
from .bot_backend import BotBackend

logger = logging.getLogger(__name__)

_UPLOAD_ERROR_TEXT: dict[str, str] = {
    "virus_detected": "failed a virus scan",
    "scan_unavailable": "couldn't be virus-scanned right now — try again shortly",
    "rejected": "was too large or would exceed your storage limit",
    "upload_failed": "couldn't be uploaded",
}


async def upload_attachments(
    api: BotBackend, ctx: MessageContext, session_id: str | None = None
) -> tuple[list[str], list[tuple[str, str]]]:
    """Upload the user's attachments to the workspace.

    ``session_id`` scopes the files to the turn's session so AutoPilot can
    read them. Returns ``(file_ids, problems)`` — the IDs that succeeded,
    and ``(filename, reason)`` for the ones that were rejected — so the
    caller can attach the successes and surface the failures.
    """
    if not ctx.attachments:
        return [], []
    try:
        results = await api.upload_workspace_files(
            platform=ctx.platform,
            platform_user_id=ctx.user_id,
            platform_server_id=ctx.server_id,
            attachments=ctx.attachments,
            session_id=session_id,
        )
    except Exception:
        # The upload path itself failed (not a per-file rejection). Don't
        # drop the message — surface it and let any text still go through.
        logger.exception("Attachment upload failed for user %s", ctx.user_id)
        return [], [(a.filename, "couldn't be uploaded") for a in ctx.attachments]
    file_ids = [r.file_id for r in results if r.file_id]
    problems = [
        (r.filename, _UPLOAD_ERROR_TEXT.get(r.error or "", "couldn't be uploaded"))
        for r in results
        if not r.file_id
    ]
    return file_ids, problems


def format_attachment_problems(problems: list[tuple[str, str]]) -> str:
    """User-facing note listing attachments that couldn't be attached and why."""
    lines = ["⚠️ Some files couldn't be attached:"]
    for filename, reason in problems:
        lines.append(f"• `{filename}` {reason}")
    return "\n".join(lines)


def model_attachment_note(problems: list[tuple[str, str]]) -> str:
    """Note injected into the turn so AutoPilot knows which files it does NOT
    have, instead of assuming a dropped attachment was read."""
    listed = ", ".join(f"{filename} ({reason})" for filename, reason in problems)
    return (
        f"[Note: the following attachment(s) could NOT be attached and are "
        f"unavailable to you — do not claim to have read them: {listed}]"
    )
