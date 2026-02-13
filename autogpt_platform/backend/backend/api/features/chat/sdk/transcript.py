"""JSONL transcript management for stateless multi-turn resume.

The Claude Code CLI persists conversations as JSONL files (one JSON object per
line).  When the SDK's ``Stop`` hook fires we read this file, store it in the DB,
and on the next turn write it back to a temp file + pass ``--resume`` so the CLI
can reconstruct the full conversation without lossy history compression.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

# Safety limit — large transcripts are truncated to keep DB writes reasonable.
MAX_TRANSCRIPT_SIZE = 512 * 1024  # 512 KB


def read_transcript_file(transcript_path: str) -> str | None:
    """Read a JSONL transcript file from disk.

    Returns the raw JSONL content, or ``None`` if the file is missing, empty,
    or only contains metadata (≤2 lines with no conversation messages).
    """
    if not transcript_path or not os.path.isfile(transcript_path):
        logger.debug(f"[Transcript] File not found: {transcript_path}")
        return None

    try:
        with open(transcript_path) as f:
            content = f.read()

        if not content.strip():
            logger.debug(f"[Transcript] Empty file: {transcript_path}")
            return None

        lines = content.strip().split("\n")
        if len(lines) < 3:
            # Metadata-only files have ≤2 lines (queue-operation + file-history-snapshot).
            logger.debug(
                f"[Transcript] Too few lines ({len(lines)}): {transcript_path}"
            )
            return None

        # Quick structural validation — parse first and last lines.
        json.loads(lines[0])
        json.loads(lines[-1])

        if len(content) > MAX_TRANSCRIPT_SIZE:
            # Truncating a JSONL transcript would break the parentUuid tree
            # structure that --resume relies on.  Instead, return None so the
            # caller falls back to the compression approach.
            logger.warning(
                f"[Transcript] Transcript too large ({len(content)} bytes), "
                "skipping — will fall back to history compression"
            )
            return None

        logger.info(
            f"[Transcript] Captured {len(lines)} lines, "
            f"{len(content)} bytes from {transcript_path}"
        )
        return content

    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[Transcript] Failed to read {transcript_path}: {e}")
        return None


def write_transcript_to_tempfile(
    transcript_content: str,
    session_id: str,
    cwd: str,
) -> str | None:
    """Write JSONL transcript to a temp file inside *cwd* for ``--resume``.

    The file lives in the session working directory so it is cleaned up
    automatically when the session ends.

    Returns the absolute path to the file, or ``None`` on failure.
    """
    try:
        os.makedirs(cwd, exist_ok=True)
        jsonl_path = os.path.join(cwd, f"transcript-{session_id[:8]}.jsonl")

        with open(jsonl_path, "w") as f:
            f.write(transcript_content)

        logger.info(f"[Transcript] Wrote resume file: {jsonl_path}")
        return jsonl_path

    except OSError as e:
        logger.warning(f"[Transcript] Failed to write resume file: {e}")
        return None


def validate_transcript(content: str | None) -> bool:
    """Check that a transcript has actual conversation messages.

    A valid transcript for resume needs at least one user message and one
    assistant message (not just queue-operation / file-history-snapshot
    metadata).
    """
    if not content or not content.strip():
        return False

    lines = content.strip().split("\n")
    if len(lines) < 3:
        return False

    has_user = False
    has_assistant = False

    for line in lines:
        try:
            entry = json.loads(line)
            msg_type = entry.get("type")
            if msg_type == "user":
                has_user = True
            elif msg_type == "assistant":
                has_assistant = True
        except json.JSONDecodeError:
            return False

    return has_user and has_assistant
