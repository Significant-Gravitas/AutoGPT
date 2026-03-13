"""JSONL transcript management for stateless multi-turn resume.

The Claude Code CLI persists conversations as JSONL files (one JSON object per
line).  When the SDK's ``Stop`` hook fires we read this file, strip bloat
(progress entries, metadata), and upload the result to bucket storage.  On the
next turn we download the transcript, write it to a temp file, and pass
``--resume`` so the CLI can reconstruct the full conversation.

Storage is handled via ``WorkspaceStorageBackend`` (GCS in prod, local
filesystem for self-hosted) — no DB column needed.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from backend.util import json

if TYPE_CHECKING:
    from backend.copilot.config import ChatConfig
    from backend.util.prompt import CompressResult

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

    Entries that are not stripped or reparented are kept as their original
    raw JSON line to avoid unnecessary re-serialization that changes
    whitespace or key ordering.
    """
    lines = content.strip().split("\n")

    # Parse entries, keeping the original line alongside the parsed dict.
    parsed: list[tuple[str, dict | None]] = []
    for line in lines:
        parsed.append((line, json.loads(line, fallback=None)))

    # First pass: identify stripped UUIDs and build parent map.
    stripped_uuids: set[str] = set()
    uuid_to_parent: dict[str, str] = {}

    for _line, entry in parsed:
        if not isinstance(entry, dict):
            continue
        uid = entry.get("uuid", "")
        parent = entry.get("parentUuid", "")
        if uid:
            uuid_to_parent[uid] = parent
        if (
            entry.get("type", "") in STRIPPABLE_TYPES
            and uid
            and not entry.get("isCompactSummary")
        ):
            stripped_uuids.add(uid)

    # Second pass: keep non-stripped entries, reparenting where needed.
    # Preserve original line when no reparenting is required.
    reparented: set[str] = set()
    for _line, entry in parsed:
        if not isinstance(entry, dict):
            continue
        parent = entry.get("parentUuid", "")
        original_parent = parent
        while parent in stripped_uuids:
            parent = uuid_to_parent.get(parent, "")
        if parent != original_parent:
            entry["parentUuid"] = parent
            uid = entry.get("uuid", "")
            if uid:
                reparented.add(uid)

    result_lines: list[str] = []
    for line, entry in parsed:
        if not isinstance(entry, dict):
            result_lines.append(line)
            continue
        if entry.get("type", "") in STRIPPABLE_TYPES and not entry.get(
            "isCompactSummary"
        ):
            continue
        uid = entry.get("uuid", "")
        if uid in reparented:
            # Re-serialize only entries whose parentUuid was changed.
            result_lines.append(json.dumps(entry, separators=(",", ":")))
        else:
            result_lines.append(line)

    return "\n".join(result_lines) + "\n"


# ---------------------------------------------------------------------------
# Local file I/O (write temp file for --resume)
# ---------------------------------------------------------------------------


def _sanitize_id(raw_id: str, max_len: int = 36) -> str:
    """Sanitize an ID for safe use in file paths.

    Session/user IDs are expected to be UUIDs (hex + hyphens).  Strip
    everything else and truncate to *max_len* so the result cannot introduce
    path separators or other special characters.
    """
    cleaned = _SAFE_ID_RE.sub("", raw_id or "")[:max_len]
    return cleaned or "unknown"


_SAFE_CWD_PREFIX = os.path.realpath("/tmp/copilot-")


def _projects_base() -> str:
    """Return the resolved path to the CLI's projects directory."""
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
    return os.path.realpath(os.path.join(config_dir, "projects"))


def _cli_project_dir(sdk_cwd: str) -> str | None:
    """Return the CLI's project directory for a given working directory.

    Returns ``None`` if the path would escape the projects base.
    """
    cwd_encoded = re.sub(r"[^a-zA-Z0-9]", "-", os.path.realpath(sdk_cwd))
    projects_base = _projects_base()
    project_dir = os.path.realpath(os.path.join(projects_base, cwd_encoded))

    if not project_dir.startswith(projects_base + os.sep):
        logger.warning(
            "[Transcript] Project dir escaped projects base: %s", project_dir
        )
        return None
    return project_dir


def _safe_glob_jsonl(project_dir: str) -> list[Path]:
    """Glob ``*.jsonl`` files, filtering out symlinks that escape the directory."""
    try:
        resolved_base = Path(project_dir).resolve()
    except OSError as e:
        logger.warning("[Transcript] Failed to resolve project dir: %s", e)
        return []

    result: list[Path] = []
    for candidate in Path(project_dir).glob("*.jsonl"):
        try:
            resolved = candidate.resolve()
            if resolved.is_relative_to(resolved_base):
                result.append(resolved)
        except (OSError, RuntimeError) as e:
            logger.debug(
                "[Transcript] Skipping invalid CLI session candidate %s: %s",
                candidate,
                e,
            )
    return result


def read_compacted_entries(transcript_path: str) -> list[dict] | None:
    """Read compacted entries from the CLI session file after compaction.

    Parses the JSONL file line-by-line, finds the ``isCompactSummary: true``
    entry, and returns it plus all entries after it.

    The CLI writes the compaction summary BEFORE sending the next message,
    so the file is guaranteed to be flushed by the time we read it.

    Returns a list of parsed dicts, or ``None`` if the file cannot be read
    or no compaction summary is found.
    """
    if not transcript_path:
        return None

    projects_base = _projects_base()
    real_path = os.path.realpath(transcript_path)
    if not real_path.startswith(projects_base + os.sep):
        logger.warning(
            "[Transcript] transcript_path outside projects base: %s", transcript_path
        )
        return None

    try:
        content = Path(real_path).read_text()
    except OSError as e:
        logger.warning(
            "[Transcript] Failed to read session file %s: %s", transcript_path, e
        )
        return None

    lines = content.strip().split("\n")
    compact_idx: int | None = None

    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        entry = json.loads(line, fallback=None)
        if not isinstance(entry, dict):
            continue
        if entry.get("isCompactSummary"):
            compact_idx = idx  # don't break — find the LAST summary

    if compact_idx is None:
        logger.debug("[Transcript] No compaction summary found in %s", transcript_path)
        return None

    entries: list[dict] = []
    for line in lines[compact_idx:]:
        if not line.strip():
            continue
        entry = json.loads(line, fallback=None)
        if isinstance(entry, dict):
            entries.append(entry)

    logger.info(
        "[Transcript] Read %d compacted entries from %s (summary at line %d)",
        len(entries),
        transcript_path,
        compact_idx + 1,
    )
    return entries


def read_cli_session_file(sdk_cwd: str) -> str | None:
    """Read the CLI's own session file, which reflects any compaction.

    The CLI writes its session transcript to
    ``~/.claude/projects/<encoded_cwd>/<session_id>.jsonl``.
    Since each SDK turn uses a unique ``sdk_cwd``, there should be
    exactly one ``.jsonl`` file in that directory.

    Returns the file content, or ``None`` if not found.
    """
    project_dir = _cli_project_dir(sdk_cwd)
    if not project_dir or not os.path.isdir(project_dir):
        return None

    jsonl_files = _safe_glob_jsonl(project_dir)
    if not jsonl_files:
        logger.debug("[Transcript] No CLI session file found in %s", project_dir)
        return None

    # Pick the most recently modified file (should be only one per turn).
    try:
        session_file = max(jsonl_files, key=lambda p: p.stat().st_mtime)
    except OSError as e:
        logger.warning("[Transcript] Failed to inspect CLI session files: %s", e)
        return None

    try:
        content = session_file.read_text()
        logger.info(
            "[Transcript] Read CLI session file: %s (%d bytes)",
            session_file,
            len(content),
        )
        return content
    except OSError as e:
        logger.warning("[Transcript] Failed to read CLI session file: %s", e)
        return None


def cleanup_cli_project_dir(sdk_cwd: str) -> None:
    """Remove the CLI's project directory for a specific working directory.

    The CLI stores session data under ``~/.claude/projects/<encoded_cwd>/``.
    Each SDK turn uses a unique ``sdk_cwd``, so the project directory is
    safe to remove entirely after the transcript has been uploaded.
    """
    project_dir = _cli_project_dir(sdk_cwd)
    if not project_dir:
        return

    if os.path.isdir(project_dir):
        shutil.rmtree(project_dir, ignore_errors=True)
        logger.debug("[Transcript] Cleaned up CLI project dir: %s", project_dir)
    else:
        logger.debug("[Transcript] Project dir not found: %s", project_dir)


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

    A valid transcript needs at least one assistant message (not just
    queue-operation / file-history-snapshot metadata).  We do NOT require
    a ``type: "user"`` entry because with ``--resume`` the user's message
    is passed as a CLI query parameter and does not appear in the
    transcript file.
    """
    if not content or not content.strip():
        return False

    lines = content.strip().split("\n")

    has_assistant = False

    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line, fallback=None)
        if not isinstance(entry, dict):
            return False
        if entry.get("type") == "assistant":
            has_assistant = True

    return has_assistant


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


def _build_path_from_parts(parts: tuple[str, str, str], backend: object) -> str:
    """Build a full storage path from (workspace_id, file_id, filename) parts."""
    from backend.util.workspace_storage import GCSWorkspaceStorage

    wid, fid, fname = parts
    if isinstance(backend, GCSWorkspaceStorage):
        blob = f"workspaces/{wid}/{fid}/{fname}"
        return f"gcs://{backend.bucket_name}/{blob}"
    return f"local://{wid}/{fid}/{fname}"


def _build_storage_path(user_id: str, session_id: str, backend: object) -> str:
    """Build the full storage path string that ``retrieve()`` expects."""
    return _build_path_from_parts(_storage_path_parts(user_id, session_id), backend)


def _build_meta_storage_path(user_id: str, session_id: str, backend: object) -> str:
    """Build the full storage path for the companion .meta.json file."""
    return _build_path_from_parts(
        _meta_storage_path_parts(user_id, session_id), backend
    )


async def upload_transcript(
    user_id: str,
    session_id: str,
    content: str,
    message_count: int = 0,
    log_prefix: str = "[Transcript]",
) -> None:
    """Strip progress entries and upload complete transcript.

    The transcript represents the FULL active context (atomic).
    Each upload REPLACES the previous transcript entirely.

    The executor holds a cluster lock per session, so concurrent uploads for
    the same session cannot happen.

    Args:
        content: Complete JSONL transcript (from TranscriptBuilder).
        message_count: ``len(session.messages)`` at upload time.
    """
    from backend.util.workspace_storage import get_workspace_storage

    # Strip metadata entries (progress, file-history-snapshot, etc.)
    # Note: SDK-built transcripts shouldn't have these, but strip for safety
    stripped = strip_progress_entries(content)
    if not validate_transcript(stripped):
        # Log entry types for debugging — helps identify why validation failed
        entry_types: list[str] = []
        for line in stripped.strip().split("\n"):
            entry = json.loads(line, fallback={"type": "INVALID_JSON"})
            entry_types.append(entry.get("type", "?"))
        logger.warning(
            "%s Skipping upload — stripped content not valid "
            "(types=%s, stripped_len=%d, raw_len=%d)",
            log_prefix,
            entry_types,
            len(stripped),
            len(content),
        )
        logger.debug("%s Raw content preview: %s", log_prefix, content[:500])
        logger.debug("%s Stripped content: %s", log_prefix, stripped[:500])
        return

    storage = await get_workspace_storage()
    wid, fid, fname = _storage_path_parts(user_id, session_id)
    encoded = stripped.encode("utf-8")

    await storage.store(
        workspace_id=wid,
        file_id=fid,
        filename=fname,
        content=encoded,
    )

    # Update metadata so message_count stays current.  The gap-fill logic
    # in _build_query_message relies on it to avoid re-compressing messages.
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
        logger.warning(f"{log_prefix} Failed to write metadata: {e}")

    logger.info(
        f"{log_prefix} Uploaded {len(encoded)}B "
        f"(stripped from {len(content)}B, msg_count={message_count})"
    )


async def download_transcript(
    user_id: str,
    session_id: str,
    log_prefix: str = "[Transcript]",
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
        logger.debug(f"{log_prefix} No transcript in storage")
        return None
    except Exception as e:
        logger.warning(f"{log_prefix} Failed to download transcript: {e}")
        return None

    # Try to load metadata (best-effort — old transcripts won't have it)
    message_count = 0
    uploaded_at = 0.0
    try:
        meta_path = _build_meta_storage_path(user_id, session_id, storage)
        meta_data = await storage.retrieve(meta_path)
        meta = json.loads(meta_data.decode("utf-8"), fallback={})
        message_count = meta.get("message_count", 0)
        uploaded_at = meta.get("uploaded_at", 0.0)
    except (FileNotFoundError, Exception):
        pass  # No metadata — treat as unknown (msg_count=0 → always fill gap)

    logger.info(f"{log_prefix} Downloaded {len(content)}B (msg_count={message_count})")
    return TranscriptDownload(
        content=content,
        message_count=message_count,
        uploaded_at=uploaded_at,
    )


# ---------------------------------------------------------------------------
# Transcript compaction
# ---------------------------------------------------------------------------

# JSONL protocol values used in transcript serialization.
STOP_REASON_END_TURN = "end_turn"
COMPACT_MSG_ID_PREFIX = "msg_compact_"
ENTRY_TYPE_MESSAGE = "message"


def _flatten_assistant_content(blocks: list) -> str:
    """Flatten assistant content blocks into a single plain-text string."""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                parts.append(f"[tool_use: {block.get('name', '?')}]")
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts) if parts else ""


def _flatten_tool_result_content(blocks: list) -> str:
    """Flatten tool_result and other content blocks into plain text.

    Handles nested tool_result structures, text blocks, and raw strings.
    Uses ``json.dumps`` as fallback for dict blocks without a ``text`` key
    or where ``text`` is ``None``.
    """
    str_parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            inner = block.get("content", "")
            if isinstance(inner, list):
                for sub in inner:
                    if isinstance(sub, dict):
                        text = sub.get("text")
                        str_parts.append(
                            str(text) if text is not None else json.dumps(sub)
                        )
                    else:
                        str_parts.append(str(sub))
            else:
                str_parts.append(str(inner))
        elif isinstance(block, dict) and block.get("type") == "text":
            str_parts.append(str(block.get("text", "")))
        elif isinstance(block, str):
            str_parts.append(block)
    return "\n".join(str_parts) if str_parts else ""


def _transcript_to_messages(content: str) -> list[dict]:
    """Convert JSONL transcript entries to message dicts for compress_context."""
    messages: list[dict] = []
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        entry = json.loads(line, fallback=None)
        if not isinstance(entry, dict):
            continue
        if entry.get("type", "") in STRIPPABLE_TYPES and not entry.get(
            "isCompactSummary"
        ):
            continue
        msg = entry.get("message", {})
        role = msg.get("role", "")
        if not role:
            continue
        msg_dict: dict = {"role": role}
        raw_content = msg.get("content")
        if role == "assistant" and isinstance(raw_content, list):
            msg_dict["content"] = _flatten_assistant_content(raw_content)
        elif isinstance(raw_content, list):
            msg_dict["content"] = _flatten_tool_result_content(raw_content)
        else:
            msg_dict["content"] = raw_content or ""
        messages.append(msg_dict)
    return messages


def _messages_to_transcript(messages: list[dict]) -> str:
    """Convert compressed message dicts back to JSONL transcript format."""
    lines: list[str] = []
    last_uuid: str | None = None
    for msg in messages:
        role = msg.get("role", "user")
        entry_type = "assistant" if role == "assistant" else "user"
        uid = str(uuid4())
        content = msg.get("content", "")
        if role == "assistant":
            message: dict = {
                "role": "assistant",
                "model": "",
                "id": f"{COMPACT_MSG_ID_PREFIX}{uuid4().hex[:24]}",
                "type": ENTRY_TYPE_MESSAGE,
                "content": [{"type": "text", "text": content}] if content else [],
                "stop_reason": STOP_REASON_END_TURN,
                "stop_sequence": None,
            }
        else:
            message = {"role": role, "content": content}
        entry = {
            "type": entry_type,
            "uuid": uid,
            "parentUuid": last_uuid,
            "message": message,
        }
        lines.append(json.dumps(entry, separators=(",", ":")))
        last_uuid = uid
    return "\n".join(lines) + "\n" if lines else ""


async def _run_compression(
    messages: list[dict],
    model: str,
    cfg: "ChatConfig",
    log_prefix: str,
) -> "CompressResult":
    """Run LLM-based compression with truncation fallback."""
    import openai

    from backend.util.prompt import compress_context

    try:
        async with openai.AsyncOpenAI(
            api_key=cfg.api_key, base_url=cfg.base_url, timeout=30.0
        ) as client:
            return await compress_context(messages=messages, model=model, client=client)
    except (openai.APIError, openai.APITimeoutError, OSError) as e:
        logger.warning("%s LLM compaction failed, using truncation: %s", log_prefix, e)
        return await compress_context(messages=messages, model=model, client=None)


async def compact_transcript(
    content: str,
    log_prefix: str = "[Transcript]",
) -> str | None:
    """Compact an oversized JSONL transcript using LLM summarization.

    Converts transcript entries to plain messages, runs ``compress_context``
    (the same compressor used for pre-query history), and rebuilds JSONL.

    Returns the compacted JSONL string, or ``None`` on failure.
    """
    from backend.copilot.config import ChatConfig

    cfg = ChatConfig()
    messages = _transcript_to_messages(content)
    if len(messages) < 2:
        logger.warning("%s Too few messages to compact (%d)", log_prefix, len(messages))
        return None
    try:
        result = await _run_compression(messages, cfg.model, cfg, log_prefix)
        if not result.was_compacted:
            logger.info("%s Transcript already within token budget", log_prefix)
            return content
        logger.info(
            "%s Compacted transcript: %d->%d tokens (%d summarized, %d dropped)",
            log_prefix,
            result.original_token_count,
            result.token_count,
            result.messages_summarized,
            result.messages_dropped,
        )
        compacted = _messages_to_transcript(result.messages)
        if not validate_transcript(compacted):
            logger.warning("%s Compacted transcript failed validation", log_prefix)
            return None
        return compacted
    except Exception as e:
        logger.error(
            "%s Transcript compaction failed: %s", log_prefix, e, exc_info=True
        )
        return None
