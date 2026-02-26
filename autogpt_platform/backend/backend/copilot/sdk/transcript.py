"""JSONL transcript management for stateless multi-turn resume.

The Claude Code CLI persists conversations as JSONL files (one JSON object per
line).  When the SDK's ``Stop`` hook fires we read this file, strip bloat
(progress entries, metadata), and upload the result to bucket storage.  On the
next turn we download the transcript, write it to a temp file, and pass
``--resume`` so the CLI can reconstruct the full conversation.

Storage is handled via ``WorkspaceStorageBackend`` (GCS in prod, local
filesystem for self-hosted) — no DB column needed.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# UUIDs are hex + hyphens; strip everything else to prevent path injection.
_SAFE_ID_RE = re.compile(r"[^0-9a-fA-F-]")

# Entry types that can be safely removed from the transcript without breaking
# the parentUuid conversation tree that ``--resume`` relies on.
# - progress: UI progress ticks, no message content (avg 97KB for agent_progress)
# - file-history-snapshot: undo tracking metadata
# - queue-operation: internal queue bookkeeping
# - summary: session summaries
# - pr-link: PR link metadata
STRIPPABLE_TYPES = frozenset(
    {"progress", "file-history-snapshot", "queue-operation", "summary", "pr-link"}
)


@dataclass
class TranscriptDownload:
    """Result of downloading a transcript with its metadata."""

    content: str
    message_count: int = 0  # session.messages length when uploaded
    uploaded_at: float = 0.0  # epoch timestamp of upload


# Workspace storage constants — deterministic path from session_id.
TRANSCRIPT_STORAGE_PREFIX = "chat-transcripts"


# ---------------------------------------------------------------------------
# Progress stripping
# ---------------------------------------------------------------------------


def strip_progress_entries(content: str) -> str:
    """Remove progress/metadata entries from a JSONL transcript.

    Removes entries whose ``type`` is in ``STRIPPABLE_TYPES`` and reparents
    any remaining child entries so the ``parentUuid`` chain stays intact.
    Typically reduces transcript size by ~30%.
    """
    lines = content.strip().split("\n")

    entries: list[dict] = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            # Keep unparseable lines as-is (safety)
            entries.append({"_raw": line})

    stripped_uuids: set[str] = set()
    uuid_to_parent: dict[str, str] = {}
    kept: list[dict] = []

    for entry in entries:
        if "_raw" in entry:
            kept.append(entry)
            continue
        uid = entry.get("uuid", "")
        parent = entry.get("parentUuid", "")
        entry_type = entry.get("type", "")

        if uid:
            uuid_to_parent[uid] = parent

        if entry_type in STRIPPABLE_TYPES:
            if uid:
                stripped_uuids.add(uid)
        else:
            kept.append(entry)

    # Reparent: walk up chain through stripped entries to find surviving ancestor
    for entry in kept:
        if "_raw" in entry:
            continue
        parent = entry.get("parentUuid", "")
        original_parent = parent
        while parent in stripped_uuids:
            parent = uuid_to_parent.get(parent, "")
        if parent != original_parent:
            entry["parentUuid"] = parent

    result_lines: list[str] = []
    for entry in kept:
        if "_raw" in entry:
            result_lines.append(entry["_raw"])
        else:
            result_lines.append(json.dumps(entry, separators=(",", ":")))

    return "\n".join(result_lines) + "\n"


# ---------------------------------------------------------------------------
# Local file I/O (read from CLI's JSONL, write temp file for --resume)
# ---------------------------------------------------------------------------


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
            logger.debug("[Transcript] File is empty: %s", transcript_path)
            return None

        lines = content.strip().split("\n")

        # Validate that the transcript has real conversation content
        # (not just metadata like queue-operation entries).
        if not validate_transcript(content):
            logger.debug(
                "[Transcript] No conversation content (%d lines) in %s",
                len(lines),
                transcript_path,
            )
            return None

        logger.info(
            f"[Transcript] Read {len(lines)} lines, "
            f"{len(content)} bytes from {transcript_path}"
        )
        return content

    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[Transcript] Failed to read {transcript_path}: {e}")
        return None


def _sanitize_id(raw_id: str, max_len: int = 36) -> str:
    """Sanitize an ID for safe use in file paths.

    Session/user IDs are expected to be UUIDs (hex + hyphens).  Strip
    everything else and truncate to *max_len* so the result cannot introduce
    path separators or other special characters.
    """
    cleaned = _SAFE_ID_RE.sub("", raw_id or "")[:max_len]
    return cleaned or "unknown"


_SAFE_CWD_PREFIX = os.path.realpath("/tmp/copilot-")


def _encode_cwd_for_cli(cwd: str) -> str:
    """Encode a working directory path the same way the Claude CLI does.

    The CLI replaces all non-alphanumeric characters with ``-``.
    """
    return re.sub(r"[^a-zA-Z0-9]", "-", os.path.realpath(cwd))


def cleanup_cli_project_dir(sdk_cwd: str) -> None:
    """Remove the CLI's project directory for a specific working directory.

    The CLI stores session data under ``~/.claude/projects/<encoded_cwd>/``.
    Each SDK turn uses a unique ``sdk_cwd``, so the project directory is
    safe to remove entirely after the transcript has been uploaded.
    """
    import shutil

    cwd_encoded = _encode_cwd_for_cli(sdk_cwd)
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
    projects_base = os.path.realpath(os.path.join(config_dir, "projects"))
    project_dir = os.path.realpath(os.path.join(projects_base, cwd_encoded))

    if not project_dir.startswith(projects_base + os.sep):
        logger.warning(
            f"[Transcript] Cleanup path escaped projects base: {project_dir}"
        )
        return

    if os.path.isdir(project_dir):
        shutil.rmtree(project_dir, ignore_errors=True)
        logger.debug(f"[Transcript] Cleaned up CLI project dir: {project_dir}")
    else:
        logger.debug(f"[Transcript] Project dir not found: {project_dir}")


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
    # Validate cwd is under the expected sandbox prefix (CodeQL sanitizer).
    real_cwd = os.path.realpath(cwd)
    if not real_cwd.startswith(_SAFE_CWD_PREFIX):
        logger.warning(f"[Transcript] cwd outside sandbox: {cwd}")
        return None

    try:
        os.makedirs(real_cwd, exist_ok=True)
        safe_id = _sanitize_id(session_id, max_len=8)
        jsonl_path = os.path.realpath(
            os.path.join(real_cwd, f"transcript-{safe_id}.jsonl")
        )
        if not jsonl_path.startswith(real_cwd):
            logger.warning(f"[Transcript] Path escaped cwd: {jsonl_path}")
            return None

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
    if len(lines) < 2:
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


# ---------------------------------------------------------------------------
# Bucket storage (GCS / local via WorkspaceStorageBackend)
# ---------------------------------------------------------------------------


def _storage_path_parts(user_id: str, session_id: str) -> tuple[str, str, str]:
    """Return (workspace_id, file_id, filename) for a session's transcript.

    Path structure: ``chat-transcripts/{user_id}/{session_id}.jsonl``
    IDs are sanitized to hex+hyphen to prevent path traversal.
    """
    return (
        TRANSCRIPT_STORAGE_PREFIX,
        _sanitize_id(user_id),
        f"{_sanitize_id(session_id)}.jsonl",
    )


def _meta_storage_path_parts(user_id: str, session_id: str) -> tuple[str, str, str]:
    """Return (workspace_id, file_id, filename) for a session's transcript metadata."""
    return (
        TRANSCRIPT_STORAGE_PREFIX,
        _sanitize_id(user_id),
        f"{_sanitize_id(session_id)}.meta.json",
    )


def _build_storage_path(user_id: str, session_id: str, backend: object) -> str:
    """Build the full storage path string that ``retrieve()`` expects.

    ``store()`` returns a path like ``gcs://bucket/workspaces/...`` or
    ``local://workspace_id/file_id/filename``.  Since we use deterministic
    arguments we can reconstruct the same path for download/delete without
    having stored the return value.
    """
    from backend.util.workspace_storage import GCSWorkspaceStorage

    wid, fid, fname = _storage_path_parts(user_id, session_id)

    if isinstance(backend, GCSWorkspaceStorage):
        blob = f"workspaces/{wid}/{fid}/{fname}"
        return f"gcs://{backend.bucket_name}/{blob}"
    else:
        # LocalWorkspaceStorage returns local://{relative_path}
        return f"local://{wid}/{fid}/{fname}"


async def upload_transcript(
    user_id: str,
    session_id: str,
    content: str,
    message_count: int = 0,
) -> None:
    """Strip progress entries and upload transcript to bucket storage.

    Safety: only overwrites when the new (stripped) transcript is larger than
    what is already stored.  Since JSONL is append-only, the latest transcript
    is always the longest.  This prevents a slow/stale background task from
    clobbering a newer upload from a concurrent turn.

    Args:
        message_count: ``len(session.messages)`` at upload time — used by
            the next turn to detect staleness and compress only the gap.
    """
    from backend.util.workspace_storage import get_workspace_storage

    stripped = strip_progress_entries(content)
    if not validate_transcript(stripped):
        logger.warning(
            f"[Transcript] Skipping upload — stripped content not valid "
            f"for session {session_id}"
        )
        return

    storage = await get_workspace_storage()
    wid, fid, fname = _storage_path_parts(user_id, session_id)
    encoded = stripped.encode("utf-8")
    new_size = len(encoded)

    # Check existing transcript size to avoid overwriting newer with older
    path = _build_storage_path(user_id, session_id, storage)
    try:
        existing = await storage.retrieve(path)
        if len(existing) >= new_size:
            logger.info(
                f"[Transcript] Skipping upload — existing ({len(existing)}B) "
                f">= new ({new_size}B) for session {session_id}"
            )
            return
    except (FileNotFoundError, Exception):
        pass  # No existing transcript or retrieval error — proceed with upload

    await storage.store(
        workspace_id=wid,
        file_id=fid,
        filename=fname,
        content=encoded,
    )

    # Store metadata alongside the transcript so the next turn can detect
    # staleness and only compress the gap instead of the full history.
    # Wrapped in try/except so a metadata write failure doesn't orphan
    # the already-uploaded transcript — the next turn will just fall back
    # to full gap fill (msg_count=0).
    try:
        meta = {"message_count": message_count, "uploaded_at": time.time()}
        mwid, mfid, mfname = _meta_storage_path_parts(user_id, session_id)
        await storage.store(
            workspace_id=mwid,
            file_id=mfid,
            filename=mfname,
            content=json.dumps(meta).encode("utf-8"),
        )
    except Exception as e:
        logger.warning(f"[Transcript] Failed to write metadata for {session_id}: {e}")

    logger.info(
        f"[Transcript] Uploaded {new_size}B "
        f"(stripped from {len(content)}B, msg_count={message_count}) "
        f"for session {session_id}"
    )


async def download_transcript(
    user_id: str, session_id: str
) -> TranscriptDownload | None:
    """Download transcript and metadata from bucket storage.

    Returns a ``TranscriptDownload`` with the JSONL content and the
    ``message_count`` watermark from the upload, or ``None`` if not found.
    """
    from backend.util.workspace_storage import get_workspace_storage

    storage = await get_workspace_storage()
    path = _build_storage_path(user_id, session_id, storage)

    try:
        data = await storage.retrieve(path)
        content = data.decode("utf-8")
    except FileNotFoundError:
        logger.debug(f"[Transcript] No transcript in storage for {session_id}")
        return None
    except Exception as e:
        logger.warning(f"[Transcript] Failed to download transcript: {e}")
        return None

    # Try to load metadata (best-effort — old transcripts won't have it)
    message_count = 0
    uploaded_at = 0.0
    try:
        from backend.util.workspace_storage import GCSWorkspaceStorage

        mwid, mfid, mfname = _meta_storage_path_parts(user_id, session_id)
        if isinstance(storage, GCSWorkspaceStorage):
            blob = f"workspaces/{mwid}/{mfid}/{mfname}"
            meta_path = f"gcs://{storage.bucket_name}/{blob}"
        else:
            meta_path = f"local://{mwid}/{mfid}/{mfname}"

        meta_data = await storage.retrieve(meta_path)
        meta = json.loads(meta_data.decode("utf-8"))
        message_count = meta.get("message_count", 0)
        uploaded_at = meta.get("uploaded_at", 0.0)
    except (FileNotFoundError, json.JSONDecodeError, Exception):
        pass  # No metadata — treat as unknown (msg_count=0 → always fill gap)

    logger.info(
        f"[Transcript] Downloaded {len(content)}B "
        f"(msg_count={message_count}) for session {session_id}"
    )
    return TranscriptDownload(
        content=content,
        message_count=message_count,
        uploaded_at=uploaded_at,
    )


async def delete_transcript(user_id: str, session_id: str) -> None:
    """Delete transcript from bucket storage (e.g. after resume failure)."""
    from backend.util.workspace_storage import get_workspace_storage

    storage = await get_workspace_storage()
    path = _build_storage_path(user_id, session_id, storage)

    try:
        await storage.delete(path)
        logger.info(f"[Transcript] Deleted transcript for session {session_id}")
    except Exception as e:
        logger.warning(f"[Transcript] Failed to delete transcript: {e}")
