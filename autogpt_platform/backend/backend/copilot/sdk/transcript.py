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

import asyncio
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from backend.util import json
from backend.util.clients import get_openai_client
from backend.util.prompt import CompressResult, compress_context
from backend.util.workspace_storage import GCSWorkspaceStorage, get_workspace_storage

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
        # seen_parents is local per-entry (not shared across iterations) so
        # it can only detect cycles within a single ancestry walk, not across
        # entries.  This is intentional: each entry's parent chain is
        # independent, and reusing a global set would incorrectly short-circuit
        # valid re-use of the same UUID as a parent in different subtrees.
        seen_parents: set[str] = set()
        while parent in stripped_uuids and parent not in seen_parents:
            seen_parents.add(parent)
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


_STALE_PROJECT_DIR_SECONDS = 12 * 3600  # 12 hours — matches max session lifetime
_MAX_PROJECT_DIRS_TO_SWEEP = 50  # limit per sweep to avoid long pauses


def cleanup_stale_project_dirs(encoded_cwd: str | None = None) -> int:
    """Remove CLI project directories older than ``_STALE_PROJECT_DIR_SECONDS``.

    Each CoPilot SDK turn creates a unique ``~/.claude/projects/<encoded-cwd>/``
    directory.  These are intentionally kept across turns so the model can read
    tool-result files via ``--resume``.  However, after a session ends they
    become stale.  This function sweeps old ones to prevent unbounded disk
    growth.

    When *encoded_cwd* is provided the sweep is scoped to that single
    directory, making the operation safe in multi-tenant environments where
    multiple copilot sessions share the same host.  Without it the function
    falls back to sweeping all directories matching the copilot naming pattern
    (``-tmp-copilot-``), which is only safe for single-tenant deployments.

    Returns the number of directories removed.
    """
    projects_base = _projects_base()
    if not os.path.isdir(projects_base):
        return 0

    now = time.time()
    removed = 0

    # Scoped mode: only clean up the one directory for the current session.
    if encoded_cwd:
        target = Path(projects_base) / encoded_cwd
        if not target.is_dir():
            return 0
        # Guard: only sweep copilot-generated dirs.
        if "-tmp-copilot-" not in target.name:
            logger.warning(
                "[Transcript] Refusing to sweep non-copilot dir: %s", target.name
            )
            return 0
        try:
            # st_mtime is used as a proxy for session activity. Claude CLI writes
            # its JSONL transcript into this directory during each turn, so mtime
            # advances on every turn. A directory whose mtime is older than
            # _STALE_PROJECT_DIR_SECONDS has not had an active turn in that window
            # and is safe to remove (the session cannot --resume after cleanup).
            age = now - target.stat().st_mtime
        except OSError:
            return 0
        if age < _STALE_PROJECT_DIR_SECONDS:
            return 0
        try:
            shutil.rmtree(target, ignore_errors=True)
            removed = 1
        except OSError:
            pass
        if removed:
            logger.info(
                "[Transcript] Swept stale CLI project dir %s (age %ds > %ds)",
                target.name,
                int(age),
                _STALE_PROJECT_DIR_SECONDS,
            )
        return removed

    # Unscoped fallback: sweep all copilot dirs across the projects base.
    # Only safe for single-tenant deployments; callers should prefer the
    # scoped variant by passing encoded_cwd.
    try:
        entries = Path(projects_base).iterdir()
    except OSError as e:
        logger.warning("[Transcript] Failed to list projects dir: %s", e)
        return 0

    for entry in entries:
        if removed >= _MAX_PROJECT_DIRS_TO_SWEEP:
            break
        # Only sweep copilot-generated dirs (pattern: -tmp-copilot- or
        # -private-tmp-copilot-).
        if "-tmp-copilot-" not in entry.name:
            continue
        if not entry.is_dir():
            continue
        try:
            # See the scoped-mode comment above: st_mtime advances on every turn,
            # so a stale mtime reliably indicates an inactive session.
            age = now - entry.stat().st_mtime
        except OSError:
            continue
        if age < _STALE_PROJECT_DIR_SECONDS:
            continue

        try:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        except OSError:
            pass

    if removed:
        logger.info(
            "[Transcript] Swept %d stale CLI project dirs (older than %ds)",
            removed,
            _STALE_PROJECT_DIR_SECONDS,
        )
    return removed


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
        logger.warning("[Transcript] cwd outside sandbox: %s", cwd)
        return None

    try:
        os.makedirs(real_cwd, exist_ok=True)
        safe_id = _sanitize_id(session_id, max_len=8)
        jsonl_path = os.path.realpath(
            os.path.join(real_cwd, f"transcript-{safe_id}.jsonl")
        )
        if not jsonl_path.startswith(real_cwd):
            logger.warning("[Transcript] Path escaped cwd: %s", jsonl_path)
            return None

        with open(jsonl_path, "w") as f:
            f.write(transcript_content)

        logger.info("[Transcript] Wrote resume file: %s", jsonl_path)
        return jsonl_path

    except OSError as e:
        logger.warning("[Transcript] Failed to write resume file: %s", e)
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
    # Strip metadata entries (progress, file-history-snapshot, etc.)
    # Note: SDK-built transcripts shouldn't have these, but strip for safety
    stripped = strip_progress_entries(content)
    if not validate_transcript(stripped):
        # Log entry types for debugging — helps identify why validation failed
        entry_types = [
            json.loads(line, fallback={"type": "INVALID_JSON"}).get("type", "?")
            for line in stripped.strip().split("\n")
        ]
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
        logger.warning("%s Failed to write metadata: %s", log_prefix, e)

    logger.info(
        "%s Uploaded %dB (stripped from %dB, msg_count=%d)",
        log_prefix,
        len(encoded),
        len(content),
        message_count,
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
    storage = await get_workspace_storage()
    path = _build_storage_path(user_id, session_id, storage)

    try:
        data = await storage.retrieve(path)
        content = data.decode("utf-8")
    except FileNotFoundError:
        logger.debug("%s No transcript in storage", log_prefix)
        return None
    except Exception as e:
        logger.warning("%s Failed to download transcript: %s", log_prefix, e)
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
    except FileNotFoundError:
        pass  # No metadata — treat as unknown (msg_count=0 → always fill gap)
    except Exception as e:
        logger.debug("%s Failed to load transcript metadata: %s", log_prefix, e)

    logger.info(
        "%s Downloaded %dB (msg_count=%d)", log_prefix, len(content), message_count
    )
    return TranscriptDownload(
        content=content,
        message_count=message_count,
        uploaded_at=uploaded_at,
    )


async def delete_transcript(user_id: str, session_id: str) -> None:
    """Delete transcript and its metadata from bucket storage.

    Removes both the ``.jsonl`` transcript and the companion ``.meta.json``
    so stale ``message_count`` watermarks cannot corrupt gap-fill logic.
    """
    storage = await get_workspace_storage()
    path = _build_storage_path(user_id, session_id, storage)

    try:
        await storage.delete(path)
        logger.info("[Transcript] Deleted transcript for session %s", session_id)
    except Exception as e:
        logger.warning("[Transcript] Failed to delete transcript: %s", e)

    # Also delete the companion .meta.json to avoid orphaned metadata.
    try:
        meta_path = _build_meta_storage_path(user_id, session_id, storage)
        await storage.delete(meta_path)
        logger.info("[Transcript] Deleted metadata for session %s", session_id)
    except Exception as e:
        logger.warning("[Transcript] Failed to delete metadata: %s", e)


# ---------------------------------------------------------------------------
# Transcript compaction — LLM summarization for prompt-too-long recovery
# ---------------------------------------------------------------------------

# JSONL protocol values used in transcript serialization.
STOP_REASON_END_TURN = "end_turn"
COMPACT_MSG_ID_PREFIX = "msg_compact_"
ENTRY_TYPE_MESSAGE = "message"


def _flatten_assistant_content(blocks: list) -> str:
    """Flatten assistant content blocks into a single plain-text string.

    Structured ``tool_use`` blocks are converted to ``[tool_use: name]``
    placeholders.  This is intentional: ``compress_context`` requires plain
    text for token counting and LLM summarization.  The structural loss is
    acceptable because compaction only runs when the original transcript was
    already too large for the model — a summarized plain-text version is
    better than no context at all.
    """
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict):
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                parts.append(f"[tool_use: {block.get('name', '?')}]")
            else:
                # Preserve non-text blocks (e.g. image) as placeholders.
                # Use __prefix__ to distinguish from literal user text.
                parts.append(f"[__{btype}__]")
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts) if parts else ""


def _flatten_tool_result_content(blocks: list) -> str:
    """Flatten tool_result and other content blocks into plain text.

    Handles nested tool_result structures, text blocks, and raw strings.
    Uses ``json.dumps`` as fallback for dict blocks without a ``text`` key
    or where ``text`` is ``None``.

    Like ``_flatten_assistant_content``, structured blocks (images, nested
    tool results) are reduced to text representations for compression.
    """
    str_parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            inner = block.get("content") or ""
            if isinstance(inner, list):
                for sub in inner:
                    if isinstance(sub, dict):
                        sub_type = sub.get("type")
                        if sub_type in ("image", "document"):
                            # Avoid serializing base64 binary data into
                            # the compaction input — use a placeholder.
                            str_parts.append(f"[__{sub_type}__]")
                        elif sub_type == "text" or sub.get("text") is not None:
                            str_parts.append(str(sub.get("text", "")))
                        else:
                            str_parts.append(json.dumps(sub))
                    else:
                        str_parts.append(str(sub))
            else:
                str_parts.append(str(inner))
        elif isinstance(block, dict) and block.get("type") == "text":
            str_parts.append(str(block.get("text", "")))
        elif isinstance(block, dict):
            # Preserve non-text/non-tool_result blocks (e.g. image) as placeholders.
            # Use __prefix__ to distinguish from literal user text.
            btype = block.get("type", "unknown")
            str_parts.append(f"[__{btype}__]")
        elif isinstance(block, str):
            str_parts.append(block)
    return "\n".join(str_parts) if str_parts else ""


def _transcript_to_messages(content: str) -> list[dict]:
    """Convert JSONL transcript entries to plain message dicts for compression.

    Parses each line of the JSONL *content*, skips strippable metadata entries
    (progress, file-history-snapshot, etc.), and extracts the ``role`` and
    flattened ``content`` from the ``message`` field of each remaining entry.

    Structured content blocks (``tool_use``, ``tool_result``, images) are
    flattened to plain text via ``_flatten_assistant_content`` and
    ``_flatten_tool_result_content`` so that ``compress_context`` can
    perform token counting and LLM summarization on uniform strings.

    Returns:
        A list of ``{"role": str, "content": str}`` dicts suitable for
        ``compress_context``.
    """
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
    """Convert compressed message dicts back to JSONL transcript format.

    Rebuilds a minimal JSONL transcript from the ``{"role", "content"}``
    dicts returned by ``compress_context``.  Each message becomes one JSONL
    line with a fresh ``uuid`` / ``parentUuid`` chain so the CLI's
    ``--resume`` flag can reconstruct a valid conversation tree.

    Assistant messages are wrapped in the full ``message`` envelope
    (``id``, ``model``, ``stop_reason``, structured ``content`` blocks)
    that the CLI expects.  User messages use the simpler ``{role, content}``
    form.

    Returns:
        A newline-terminated JSONL string, or an empty string if *messages*
        is empty.
    """
    lines: list[str] = []
    last_uuid: str = ""  # root entry uses empty string, not null
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


_COMPACTION_TIMEOUT_SECONDS = 60
_TRUNCATION_TIMEOUT_SECONDS = 30


async def _run_compression(
    messages: list[dict],
    model: str,
    log_prefix: str,
) -> CompressResult:
    """Run LLM-based compression with truncation fallback.

    Uses the shared OpenAI client from ``get_openai_client()``.
    If no client is configured or the LLM call fails, falls back to
    truncation-based compression which drops older messages without
    summarization.

    A 60-second timeout prevents a hung LLM call from blocking the
    retry path indefinitely.  The truncation fallback also has a
    30-second timeout to guard against slow tokenization on very large
    transcripts.
    """
    client = get_openai_client()
    if client is None:
        logger.warning("%s No OpenAI client configured, using truncation", log_prefix)
        return await asyncio.wait_for(
            compress_context(messages=messages, model=model, client=None),
            timeout=_TRUNCATION_TIMEOUT_SECONDS,
        )
    try:
        return await asyncio.wait_for(
            compress_context(messages=messages, model=model, client=client),
            timeout=_COMPACTION_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.warning("%s LLM compaction failed, using truncation: %s", log_prefix, e)
        return await asyncio.wait_for(
            compress_context(messages=messages, model=model, client=None),
            timeout=_TRUNCATION_TIMEOUT_SECONDS,
        )


async def compact_transcript(
    content: str,
    *,
    model: str,
    log_prefix: str = "[Transcript]",
) -> str | None:
    """Compact an oversized JSONL transcript using LLM summarization.

    Converts transcript entries to plain messages, runs ``compress_context``
    (the same compressor used for pre-query history), and rebuilds JSONL.

    Structured content (``tool_use`` blocks, ``tool_result`` nesting, images)
    is flattened to plain text for compression.  This matches the fidelity of
    the Plan C (DB compression) fallback path, where
    ``_format_conversation_context`` similarly renders tool calls as
    ``You called tool: name(args)`` and results as ``Tool result: ...``.
    Neither path preserves structured API content blocks — the compacted
    context serves as text history for the LLM, which creates proper
    structured tool calls going forward.

    Images are per-turn attachments loaded from workspace storage by file ID
    (via ``_prepare_file_attachments``), not part of the conversation history.
    They are re-attached each turn and are unaffected by compaction.

    Returns the compacted JSONL string, or ``None`` on failure.

    See also:
        ``_compress_messages`` in ``service.py`` — compresses ``ChatMessage``
        lists for pre-query DB history.  Both share ``compress_context()``
        but operate on different input formats (JSONL transcript entries
        here vs. ChatMessage dicts there).
    """
    messages = _transcript_to_messages(content)
    if len(messages) < 2:
        logger.warning("%s Too few messages to compact (%d)", log_prefix, len(messages))
        return None
    try:
        result = await _run_compression(messages, model, log_prefix)
        if not result.was_compacted:
            # Compressor says it's within budget, but the SDK rejected it.
            # Return None so the caller falls through to DB fallback.
            logger.warning(
                "%s Compressor reports within budget but SDK rejected — "
                "signalling failure",
                log_prefix,
            )
            return None
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
