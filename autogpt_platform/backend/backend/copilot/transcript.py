"""JSONL transcript management for stateless multi-turn resume.

The Claude Code CLI persists conversations as JSONL files (one JSON object per
line).  When the SDK's ``Stop`` hook fires the caller reads this file, strips
bloat (progress entries, metadata), and uploads the result to bucket storage.
On the next turn the caller downloads the bytes and writes them to disk before
passing ``--resume`` so the CLI can reconstruct the full conversation.

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
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from backend.util import json
from backend.util.clients import get_openai_client
from backend.util.prompt import CompressResult, compress_context
from backend.util.workspace_storage import GCSWorkspaceStorage, get_workspace_storage

if TYPE_CHECKING:
    from .model import ChatMessage

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


TranscriptMode = Literal["sdk", "baseline"]


@dataclass
class TranscriptDownload:
    content: bytes | str
    message_count: int = 0
    # "sdk" = Claude CLI native, "baseline" = TranscriptBuilder
    mode: TranscriptMode = "sdk"


# Storage prefix for the CLI's native session JSONL files (for cross-pod --resume).
_CLI_SESSION_STORAGE_PREFIX = "cli-sessions"


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


def strip_stale_thinking_blocks(content: str) -> str:
    """Remove thinking/redacted_thinking blocks from non-last assistant entries.

    The Anthropic API only requires thinking blocks in the **last** assistant
    message to be value-identical to the original response.  Older assistant
    entries carry stale thinking blocks that consume significant tokens
    (often 10-50K each) without providing useful context for ``--resume``.

    Stripping them before upload prevents the CLI from triggering compaction
    every turn just to compress away the stale thinking bloat.
    """
    lines = content.strip().split("\n")
    if not lines:
        return content

    parsed: list[tuple[str, dict | None]] = []
    for line in lines:
        parsed.append((line, json.loads(line, fallback=None)))

    # Reverse scan to find the last assistant message ID and index.
    last_asst_msg_id: str | None = None
    last_asst_idx: int | None = None
    for i in range(len(parsed) - 1, -1, -1):
        _line, entry = parsed[i]
        if not isinstance(entry, dict):
            continue
        msg = entry.get("message", {})
        if msg.get("role") == "assistant":
            last_asst_msg_id = msg.get("id")
            last_asst_idx = i
            break

    if last_asst_idx is None:
        return content

    result_lines: list[str] = []
    stripped_count = 0
    for i, (line, entry) in enumerate(parsed):
        if not isinstance(entry, dict):
            result_lines.append(line)
            continue

        msg = entry.get("message", {})
        # Only strip from assistant entries that are NOT the last turn.
        # Use msg_id matching when available; fall back to index for entries
        # without an id field.
        is_last_turn = (
            last_asst_msg_id is not None and msg.get("id") == last_asst_msg_id
        ) or (last_asst_msg_id is None and i == last_asst_idx)
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            content_blocks = msg["content"]
            producing_model = msg.get("model") if isinstance(msg, dict) else None
            filtered = [
                b
                for b in content_blocks
                if not _should_strip_thinking_block(
                    b,
                    is_last_turn=is_last_turn,
                    producing_model=producing_model,
                )
            ]
            if len(filtered) < len(content_blocks):
                stripped_count += len(content_blocks) - len(filtered)
                entry = {**entry, "message": {**msg, "content": filtered}}
                result_lines.append(json.dumps(entry, separators=(",", ":")))
                continue

        result_lines.append(line)

    if stripped_count:
        logger.info(
            "[Transcript] Stripped %d stale thinking block(s) from non-last entries",
            stripped_count,
        )

    return "\n".join(result_lines) + "\n"


def strip_for_upload(content: str) -> str:
    """Combined single-parse strip of progress entries and stale thinking blocks.

    Equivalent to ``strip_stale_thinking_blocks(strip_progress_entries(content))``
    but parses the JSONL only once, avoiding redundant ``split`` + ``json.loads``
    passes on every upload.
    """
    lines = content.strip().split("\n")
    if not lines:
        return content

    parsed: list[tuple[str, dict | None]] = []
    for line in lines:
        parsed.append((line, json.loads(line, fallback=None)))

    # --- Phase 1: progress stripping (reparent children) ---
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

    reparented: set[str] = set()
    for _line, entry in parsed:
        if not isinstance(entry, dict):
            continue
        parent = entry.get("parentUuid", "")
        original_parent = parent
        seen_parents: set[str] = set()
        while parent in stripped_uuids and parent not in seen_parents:
            seen_parents.add(parent)
            parent = uuid_to_parent.get(parent, "")
        if parent != original_parent:
            entry["parentUuid"] = parent
            uid = entry.get("uuid", "")
            if uid:
                reparented.add(uid)

    # --- Phase 2: identify last assistant for thinking-block stripping ---
    last_asst_msg_id: str | None = None
    last_asst_idx: int | None = None
    for i in range(len(parsed) - 1, -1, -1):
        _line, entry = parsed[i]
        if not isinstance(entry, dict):
            continue
        if entry.get("type", "") in STRIPPABLE_TYPES and not entry.get(
            "isCompactSummary"
        ):
            continue
        msg = entry.get("message", {})
        if msg.get("role") == "assistant":
            last_asst_msg_id = msg.get("id")
            last_asst_idx = i
            break

    # --- Phase 3: single output pass ---
    result_lines: list[str] = []
    thinking_stripped = 0
    for i, (line, entry) in enumerate(parsed):
        if not isinstance(entry, dict):
            result_lines.append(line)
            continue

        # Drop progress/metadata entries
        if entry.get("type", "") in STRIPPABLE_TYPES and not entry.get(
            "isCompactSummary"
        ):
            continue

        needs_reserialize = False
        uid = entry.get("uuid", "")

        # Reparented entries need re-serialization
        if uid in reparented:
            needs_reserialize = True

        # Strip stale thinking blocks from non-last assistant entries.
        # Also strip *signature-less* thinking blocks from the last entry —
        # those come from non-Anthropic providers (e.g. Kimi K2.6 via
        # OpenRouter) and are rejected with ``Invalid `signature` in
        # `thinking` block`` if a subsequent turn is dispatched to an
        # Anthropic model that re-validates them.  Anthropic-emitted
        # thinking blocks always carry a non-empty ``signature`` field, so
        # this filter is a no-op on Sonnet/Opus turns and only kicks in
        # when the prior turn ran on a non-Anthropic vendor.
        if last_asst_idx is not None:
            msg = entry.get("message", {})
            is_last_turn = (
                last_asst_msg_id is not None and msg.get("id") == last_asst_msg_id
            ) or (last_asst_msg_id is None and i == last_asst_idx)
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                content_blocks = msg["content"]
                producing_model = msg.get("model") if isinstance(msg, dict) else None
                filtered = [
                    b
                    for b in content_blocks
                    if not _should_strip_thinking_block(
                        b,
                        is_last_turn=is_last_turn,
                        producing_model=producing_model,
                    )
                ]
                if len(filtered) < len(content_blocks):
                    thinking_stripped += len(content_blocks) - len(filtered)
                    entry = {**entry, "message": {**msg, "content": filtered}}
                    needs_reserialize = True

        if needs_reserialize:
            result_lines.append(json.dumps(entry, separators=(",", ":")))
        else:
            result_lines.append(line)

    if thinking_stripped:
        logger.info(
            "[Transcript] Stripped %d stale thinking block(s) from non-last entries",
            thinking_stripped,
        )

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


def projects_base() -> str:
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
    _pbase = projects_base()
    if not os.path.isdir(_pbase):
        return 0

    now = time.time()
    removed = 0

    # Scoped mode: only clean up the one directory for the current session.
    if encoded_cwd:
        target = Path(_pbase) / encoded_cwd
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
        entries = Path(_pbase).iterdir()
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

    _pbase = projects_base()
    real_path = os.path.realpath(transcript_path)
    if not real_path.startswith(_pbase + os.sep):
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


def _build_path_from_parts(parts: tuple[str, str, str], backend: object) -> str:
    """Build a full storage path from (workspace_id, file_id, filename) parts."""
    wid, fid, fname = parts
    if isinstance(backend, GCSWorkspaceStorage):
        blob = f"workspaces/{wid}/{fid}/{fname}"
        return f"gcs://{backend.bucket_name}/{blob}"
    return f"local://{wid}/{fid}/{fname}"


# ---------------------------------------------------------------------------
# CLI native session file — cross-pod --resume support
# ---------------------------------------------------------------------------


def cli_session_path(sdk_cwd: str, session_id: str) -> str:
    """Expected path of the CLI's native session JSONL file.

    The CLI resolves the working directory via ``os.path.realpath``, then
    encodes it by replacing every non-alphanumeric character with ``-``,
    placing its session file at::

        {projects_base}/{encoded_cwd}/{session_id}.jsonl

    We must mirror the CLI's realpath + regex encoding exactly.  On macOS
    ``/tmp`` is a symlink to ``/private/tmp``, so a naive ``str.replace("/",
    "-")`` would produce the wrong directory name and the file would never be
    found.
    """
    encoded_cwd = re.sub(r"[^a-zA-Z0-9]", "-", os.path.realpath(sdk_cwd))
    safe_id = _sanitize_id(session_id)
    return os.path.join(projects_base(), encoded_cwd, f"{safe_id}.jsonl")


def _cli_session_storage_path_parts(
    user_id: str, session_id: str
) -> tuple[str, str, str]:
    """Return (workspace_id, file_id, filename) for a CLI session file in storage."""
    return (
        _CLI_SESSION_STORAGE_PREFIX,
        _sanitize_id(user_id),
        f"{_sanitize_id(session_id)}.jsonl",
    )


def _cli_session_meta_path_parts(user_id: str, session_id: str) -> tuple[str, str, str]:
    """Return (workspace_id, file_id, filename) for the CLI session meta file."""
    return (
        _CLI_SESSION_STORAGE_PREFIX,
        _sanitize_id(user_id),
        f"{_sanitize_id(session_id)}.meta.json",
    )


async def upload_transcript(
    user_id: str,
    session_id: str,
    content: bytes,
    message_count: int = 0,
    mode: TranscriptMode = "sdk",
    log_prefix: str = "[Transcript]",
) -> None:
    """Upload CLI session content to GCS with companion meta.json.

    Pure GCS operation — no disk I/O.  The caller is responsible for reading
    the session file from disk before calling this function.

    Also uploads a companion .meta.json with the message_count watermark so
    download_transcript can return it without a separate fetch.

    Called after each turn so the next turn can restore the file on any pod
    (eliminating the pod-affinity requirement for --resume).
    """
    storage = await get_workspace_storage()
    wid, fid, fname = _cli_session_storage_path_parts(user_id, session_id)
    mwid, mfid, mfname = _cli_session_meta_path_parts(user_id, session_id)
    meta = {"message_count": message_count, "mode": mode, "uploaded_at": time.time()}
    meta_encoded = json.dumps(meta).encode("utf-8")

    # Write JSONL first, meta second — sequential so a crash between the two
    # leaves an orphaned JSONL (no meta) rather than an orphaned meta (wrong
    # watermark / mode paired with stale or absent content).
    # On any failure we roll back the other file so the pair is always absent
    # together; download_transcript returns None when either file is missing.
    try:
        await storage.store(
            workspace_id=wid, file_id=fid, filename=fname, content=content
        )
    except Exception as session_err:
        logger.warning(
            "%s Failed to upload CLI session file: %s", log_prefix, session_err
        )
        return

    try:
        await storage.store(
            workspace_id=mwid, file_id=mfid, filename=mfname, content=meta_encoded
        )
    except Exception as meta_err:
        logger.warning("%s Failed to upload CLI session meta: %s", log_prefix, meta_err)
        # Roll back the JSONL so neither file exists — avoids orphaned JSONL being
        # used with wrong mode/watermark defaults on the next restore.
        try:
            session_path = _build_path_from_parts(
                _cli_session_storage_path_parts(user_id, session_id), storage
            )
            await storage.delete(session_path)
        except Exception as rollback_err:
            logger.debug(
                "%s Session rollback failed (harmless — download will return None): %s",
                log_prefix,
                rollback_err,
            )
        return

    logger.info(
        "%s Uploaded CLI session (%dB, msg_count=%d, mode=%s)",
        log_prefix,
        len(content),
        message_count,
        mode,
    )


async def download_transcript(
    user_id: str,
    session_id: str,
    log_prefix: str = "[Transcript]",
) -> TranscriptDownload | None:
    """Download CLI session from GCS. Returns content + message_count + mode, or None if not found.

    Pure GCS operation — no disk I/O.  The caller is responsible for writing
    content to disk if --resume is needed.

    Returns a TranscriptDownload with the raw content, message_count watermark,
    and mode on success, or None if not available (first turn or upload failed).
    """
    storage = await get_workspace_storage()
    path = _build_path_from_parts(
        _cli_session_storage_path_parts(user_id, session_id), storage
    )
    meta_path = _build_path_from_parts(
        _cli_session_meta_path_parts(user_id, session_id), storage
    )

    content_result, meta_result = await asyncio.gather(
        storage.retrieve(path),
        storage.retrieve(meta_path),
        return_exceptions=True,
    )

    if isinstance(content_result, FileNotFoundError):
        logger.debug("%s No CLI session in storage (first turn or missing)", log_prefix)
        return None
    if isinstance(content_result, BaseException):
        logger.warning(
            "%s Failed to download CLI session: %s", log_prefix, content_result
        )
        return None

    content: bytes = content_result

    # Parse message_count and mode from companion meta — best-effort, defaults.
    message_count = 0
    mode: TranscriptMode = "sdk"
    if isinstance(meta_result, FileNotFoundError):
        pass  # No meta — old upload; default to "sdk"
    elif isinstance(meta_result, BaseException):
        logger.debug("%s Failed to load CLI session meta: %s", log_prefix, meta_result)
    else:
        try:
            meta_str = meta_result.decode("utf-8")
        except UnicodeDecodeError:
            logger.debug("%s CLI session meta is not valid UTF-8, ignoring", log_prefix)
            meta_str = None
        if meta_str is not None:
            meta = json.loads(meta_str, fallback={})
            if isinstance(meta, dict):
                raw_count = meta.get("message_count", 0)
                message_count = (
                    raw_count if isinstance(raw_count, int) and raw_count >= 0 else 0
                )
                raw_mode = meta.get("mode", "sdk")
                mode = raw_mode if raw_mode in ("sdk", "baseline") else "sdk"

    logger.info(
        "%s Downloaded CLI session (%dB, msg_count=%d, mode=%s)",
        log_prefix,
        len(content),
        message_count,
        mode,
    )
    return TranscriptDownload(content=content, message_count=message_count, mode=mode)


def detect_gap(
    download: TranscriptDownload,
    session_messages: list[ChatMessage],
) -> list[ChatMessage]:
    """Return chat-db messages after the transcript watermark (excluding current user turn).

    Returns [] if transcript is current, watermark is zero, or the watermark
    position doesn't end on an assistant turn (misaligned watermark).
    """
    if download.message_count == 0:
        return []
    wm = download.message_count
    total = len(session_messages)
    if wm >= total - 1:
        return []
    # Sanity: position wm-1 should be an assistant turn; misaligned watermark
    # means the DB messages shifted (e.g. deletion) — skip gap to avoid wrong context.
    # In normal operation ``message_count`` is always written after a complete
    # user→assistant exchange (never mid-turn), so the last covered position is
    # always assistant.  This guard fires only on data corruption or message deletion.
    if session_messages[wm - 1].role != "assistant":
        return []
    return list(session_messages[wm : total - 1])


def extract_context_messages(
    download: TranscriptDownload | None,
    session_messages: "list[ChatMessage]",
) -> "list[ChatMessage]":
    """Return context messages for the current turn: transcript content + gap.

    This is the shared context primitive used by both the SDK path
    (``use_resume=False`` → ``<conversation_history>`` injection) and the
    baseline path (OpenAI messages array).

    How it works:

    - When a transcript exists, ``TranscriptBuilder.load_previous`` preserves
      ``isCompactSummary=True`` compaction entries, so the returned messages
      mirror the compacted context the CLI would see via ``--resume``.
    - The gap (DB messages after the transcript watermark) is always small in
      normal operation; it only grows during mode switches or when an upload
      was missed.
    - Falls back to full DB messages when no transcript exists (first turn,
      upload failure, or GCS unavailable).
    - Returns *prior* messages only (excluding the current user turn at
      ``session_messages[-1]``).  Callers that need the current turn append
      ``session_messages[-1]`` themselves.
    - **Tool calls from transcript entries are flattened to text**: assistant
      messages derived from the JSONL use ``_flatten_assistant_content``, which
      serialises ``tool_use`` blocks as human-readable text rather than
      structured ``tool_calls``.  Gap messages (from DB) preserve their
      original ``tool_calls`` field.  This is the same trade-off as the old
      ``_compress_session_messages(session.messages)`` approach — no regression.

    Args:
        download: The ``TranscriptDownload`` from GCS, or ``None`` when no
            transcript is available.  ``content`` may be either ``bytes`` or
            ``str`` (the baseline path decodes + strips before returning).
        session_messages: All messages in the session, with the current user
            turn as the last element.

    Returns:
        A list of ``ChatMessage`` objects covering the prior conversation
        context, suitable for injection as conversation history.
    """
    from .model import ChatMessage as _ChatMessage  # runtime import

    # ``role="reasoning"`` rows are persisted for frontend replay of
    # extended_thinking content but are NOT conversation context — the
    # transcript-based --resume path already carries thinking separately,
    # and sending them back to the model as user/assistant turns would be
    # both redundant and malformed.  Drop them before any gap detection
    # or transcript comparison so ordering invariants still hold.
    session_messages = [m for m in session_messages if m.role != "reasoning"]

    prior = session_messages[:-1]

    if download is None:
        return prior

    raw_content = download.content
    if not raw_content:
        return prior

    # Handle both bytes (raw GCS download) and str (pre-decoded baseline path).
    if isinstance(raw_content, bytes):
        try:
            content_str: str = raw_content.decode("utf-8")
        except UnicodeDecodeError:
            return prior
    else:
        content_str = raw_content

    raw = _transcript_to_messages(content_str)
    if not raw:
        return prior

    transcript_msgs = [
        _ChatMessage(role=m["role"], content=m.get("content") or "") for m in raw
    ]
    gap = detect_gap(download, session_messages)
    return transcript_msgs + gap


async def delete_transcript(user_id: str, session_id: str) -> None:
    """Delete CLI session JSONL and its companion .meta.json from bucket storage."""
    storage = await get_workspace_storage()

    try:
        cli_path = _build_path_from_parts(
            _cli_session_storage_path_parts(user_id, session_id), storage
        )
        await storage.delete(cli_path)
        logger.info("[Transcript] Deleted CLI session for session %s", session_id)
    except Exception as e:
        logger.warning("[Transcript] Failed to delete CLI session: %s", e)

    try:
        cli_meta_path = _build_path_from_parts(
            _cli_session_meta_path_parts(user_id, session_id), storage
        )
        await storage.delete(cli_meta_path)
        logger.info("[Transcript] Deleted CLI session meta for session %s", session_id)
    except Exception as e:
        logger.warning("[Transcript] Failed to delete CLI session meta: %s", e)


# ---------------------------------------------------------------------------
# Transcript compaction — LLM summarization for prompt-too-long recovery
# ---------------------------------------------------------------------------

# JSONL protocol values used in transcript serialization.
STOP_REASON_END_TURN = "end_turn"
STOP_REASON_TOOL_USE = "tool_use"
COMPACT_MSG_ID_PREFIX = "msg_compact_"
ENTRY_TYPE_MESSAGE = "message"


_THINKING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})


def _is_anthropic_model(model: str | None) -> bool:
    """True when *model* is an Anthropic-issued slug.

    Used to decide whether a thinking block's signature is
    cryptographically valid for Anthropic replay.  Non-Anthropic vendors
    routed through OpenRouter's Anthropic-compat shim (Kimi K2.6,
    DeepSeek, GPT-OSS) sometimes emit thinking blocks with a
    placeholder signature — it passes a non-empty string check but
    fails Anthropic's cryptographic validation, producing the opaque
    ``Invalid signature in thinking block`` 400 on the next turn
    whenever the model toggle switches to Sonnet/Opus.
    """
    return isinstance(model, str) and model.startswith("anthropic/")


def _should_strip_thinking_block(
    block: object,
    *,
    is_last_turn: bool,
    producing_model: str | None = None,
) -> bool:
    """Return True when *block* is a thinking block that should be removed
    from a transcript entry before upload.

    Strip only when the block CAN'T be replayed safely.  Never strip a
    valid Anthropic-issued thinking block — it carries real reasoning
    state that preserves context continuity on ``--resume``.

    Strip rules (first match wins):

    1. **Non-Anthropic producer (any position)** — thinking blocks from
       Kimi / DeepSeek / GPT-OSS via OpenRouter's Anthropic-compat shim
       carry either no signature or a placeholder string that passes a
       non-empty check but fails Anthropic's cryptographic validation.
       Strip unconditionally; they also add low-value tokens to the
       replay context.
    2. **Malformed ``thinking`` (any position, Anthropic producer,
       empty signature)** — shouldn't happen in practice, but if the
       signature is missing / empty the block can't be validated.
       Safer to drop than to 400 the next turn.
    3. **Stale non-last entry with unknown producer** — when the
       caller doesn't wire ``producing_model`` through (legacy paths /
       older tests) we can't tell if the block is safe to keep; fall
       back to the old behaviour of dropping non-last thinking blocks
       to avoid replaying an unverifiable block to Anthropic.

    Preserved:

    * Anthropic ``thinking`` with non-empty signature — at any
      position, last OR non-last.  Keeping prior-turn reasoning
      chains helps continuity on multi-round SDK resumes without any
      risk of signature rejection.
    * Anthropic ``redacted_thinking`` — carries an encrypted ``data``
      payload instead of a ``signature``; by design signature-less,
      but Anthropic-issued and safely replayable.
    """
    if not isinstance(block, dict):
        return False
    btype = block.get("type")
    if btype not in _THINKING_BLOCK_TYPES:
        return False
    # Legacy call sites pass producing_model=None — preserve the old
    # "strip-all-non-last-thinking" heuristic for those so we don't
    # regress callers that haven't been updated.
    if producing_model is None:
        if not is_last_turn:
            return True
        if btype != "thinking":
            return False
        signature = block.get("signature")
        return not (isinstance(signature, str) and signature)
    # Non-Anthropic producer — strip at any position.  These blocks
    # CAN'T be cryptographically validated by Anthropic on replay.
    if not _is_anthropic_model(producing_model):
        return True
    # Anthropic producer, redacted_thinking: always preserve — the
    # ``data`` field is the signature analog.
    if btype == "redacted_thinking":
        return False
    # Anthropic producer, ``thinking``: keep iff it has a real
    # (non-empty) signature.  Empty-signature Anthropic thinking
    # shouldn't happen but guard against it anyway.
    signature = block.get("signature")
    return not (isinstance(signature, str) and signature)


def _flatten_assistant_content(blocks: list) -> str:
    """Flatten assistant content blocks into a single plain-text string.

    Structured ``tool_use`` blocks are converted to ``[tool_use: name]``
    placeholders.  ``thinking`` and ``redacted_thinking`` blocks are
    silently dropped — they carry no useful context for compression
    summaries and must not leak into compacted transcripts (the Anthropic
    API requires thinking blocks in the last assistant message to be
    value-identical to the original response; including stale thinking
    text would violate that constraint).

    This is intentional: ``compress_context`` requires plain text for
    token counting and LLM summarization.  The structural loss is
    acceptable because compaction only runs when the original transcript
    was already too large for the model.
    """
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict):
            btype = block.get("type", "")
            if btype in _THINKING_BLOCK_TYPES:
                continue
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                # Drop tool_use entirely — any text representation gets
                # mimicked by the model as plain text instead of actual
                # structured tool calls. The tool results (in the
                # following user/tool_result entry) provide sufficient
                # context about what happened.
                continue
            else:
                continue
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
    target_tokens: int | None = None,
) -> CompressResult:
    """Run LLM-based compression with truncation fallback.

    Uses the shared OpenAI client from ``get_openai_client()``.
    If no client is configured or the LLM call fails, falls back to
    truncation-based compression which drops older messages without
    summarization.

    ``target_tokens`` sets a hard token ceiling for the compressed output.
    When ``None``, ``compress_context`` derives the limit from the model's
    context window.  Pass a smaller value on retries to force more aggressive
    compression — the compressor will LLM-summarize, content-truncate,
    middle-out delete, and first/last trim until the result fits.

    A 60-second timeout prevents a hung LLM call from blocking the
    retry path indefinitely.  The truncation fallback also has a
    30-second timeout to guard against slow tokenization on very large
    transcripts.
    """
    client = get_openai_client()
    if client is None:
        logger.warning("%s No OpenAI client configured, using truncation", log_prefix)
        return await asyncio.wait_for(
            compress_context(
                messages=messages, model=model, client=None, target_tokens=target_tokens
            ),
            timeout=_TRUNCATION_TIMEOUT_SECONDS,
        )
    try:
        return await asyncio.wait_for(
            compress_context(
                messages=messages,
                model=model,
                client=client,
                target_tokens=target_tokens,
            ),
            timeout=_COMPACTION_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.warning("%s LLM compaction failed, using truncation: %s", log_prefix, e)
        return await asyncio.wait_for(
            compress_context(
                messages=messages, model=model, client=None, target_tokens=target_tokens
            ),
            timeout=_TRUNCATION_TIMEOUT_SECONDS,
        )


def _find_last_assistant_entry(
    content: str,
) -> tuple[list[str], list[str]]:
    """Split JSONL lines into (compressible_prefix, preserved_tail).

    The tail starts at the **first** entry of the last assistant turn and
    includes everything after it (typically trailing user messages).  An
    assistant turn can span multiple consecutive JSONL entries sharing the
    same ``message.id`` (e.g., a thinking entry followed by a tool_use
    entry).  All entries of the turn are preserved verbatim.

    The Anthropic API requires that ``thinking`` and ``redacted_thinking``
    blocks in the **last** assistant message remain value-identical to the
    original response (the API validates parsed signature values, not raw
    JSON bytes).  By excluding the entire turn from compression we
    guarantee those blocks are never altered.

    Returns ``(all_lines, [])`` when no assistant entry is found.
    """
    lines = [ln for ln in content.strip().split("\n") if ln.strip()]

    # Parse all lines once to avoid double JSON deserialization.
    # json.loads with fallback=None returns Any; non-dict entries are
    # safely skipped by the isinstance(entry, dict) guards below.
    parsed: list = [json.loads(ln, fallback=None) for ln in lines]

    # Reverse scan: find the message.id and index of the last assistant entry.
    last_asst_msg_id: str | None = None
    last_asst_idx: int | None = None
    for i in range(len(parsed) - 1, -1, -1):
        entry = parsed[i]
        if not isinstance(entry, dict):
            continue
        msg = entry.get("message", {})
        if msg.get("role") == "assistant":
            last_asst_idx = i
            last_asst_msg_id = msg.get("id")
            break

    if last_asst_idx is None:
        return lines, []

    # If the assistant entry has no message.id, fall back to preserving
    # from that single entry onward — safer than compressing everything.
    if last_asst_msg_id is None:
        return lines[:last_asst_idx], lines[last_asst_idx:]

    # Forward scan: find the first entry of this turn (same message.id).
    first_turn_idx: int | None = None
    for i, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            continue
        msg = entry.get("message", {})
        if msg.get("role") == "assistant" and msg.get("id") == last_asst_msg_id:
            first_turn_idx = i
            break

    if first_turn_idx is None:
        return lines, []
    return lines[:first_turn_idx], lines[first_turn_idx:]


async def compact_transcript(
    content: str,
    *,
    model: str,
    log_prefix: str = "[Transcript]",
    target_tokens: int | None = None,
) -> str | None:
    """Compact an oversized JSONL transcript using LLM summarization.

    Converts transcript entries to plain messages, runs ``compress_context``
    (the same compressor used for pre-query history), and rebuilds JSONL.

    The **last assistant entry** (and any entries after it) are preserved
    verbatim — never flattened or compressed.  The Anthropic API requires
    ``thinking`` and ``redacted_thinking`` blocks in the latest assistant
    message to be value-identical to the original response (the API
    validates parsed signature values, not raw JSON bytes); compressing
    them would destroy the cryptographic signatures and cause
    ``invalid_request_error``.

    Structured content in *older* assistant entries (``tool_use`` blocks,
    ``thinking`` blocks, ``tool_result`` nesting, images) is flattened to
    plain text for compression.  This matches the fidelity of the Plan C
    (DB compression) fallback path.

    Returns the compacted JSONL string, or ``None`` on failure.

    See also:
        ``_compress_messages`` in ``service.py`` — compresses ``ChatMessage``
        lists for pre-query DB history.
    """
    prefix_lines, tail_lines = _find_last_assistant_entry(content)

    # Build the JSONL string for the compressible prefix
    prefix_content = "\n".join(prefix_lines) + "\n" if prefix_lines else ""
    messages = _transcript_to_messages(prefix_content) if prefix_content else []

    if len(messages) + len(tail_lines) < 2:
        total = len(messages) + len(tail_lines)
        logger.warning("%s Too few messages to compact (%d)", log_prefix, total)
        return None
    if not messages:
        logger.warning("%s Nothing to compress (only tail entries remain)", log_prefix)
        return None
    try:
        result = await _run_compression(
            messages, model, log_prefix, target_tokens=target_tokens
        )
        if not result.was_compacted:
            logger.warning(
                "%s Compressor reports within budget but SDK rejected — "
                "signalling failure",
                log_prefix,
            )
            return None
        if not result.messages:
            logger.warning("%s Compressor returned empty messages", log_prefix)
            return None
        logger.info(
            "%s Compacted transcript: %d->%d tokens (%d summarized, %d dropped)",
            log_prefix,
            result.original_token_count,
            result.token_count,
            result.messages_summarized,
            result.messages_dropped,
        )
        compressed_part = _messages_to_transcript(result.messages)

        # Re-append the preserved tail (last assistant + trailing entries)
        # with parentUuid patched to chain onto the compressed prefix.
        tail_part = _rechain_tail(compressed_part, tail_lines)
        compacted = compressed_part + tail_part

        if len(compacted) >= len(content):
            # Byte count can increase due to preserved tail entries
            # (thinking blocks, JSON overhead) even when token count
            # decreased.  Log a warning but still return — the API
            # validates tokens not bytes, and the caller falls through
            # to DB fallback if the transcript is still too large.
            logger.warning(
                "%s Compacted transcript (%d bytes) is not smaller than "
                "original (%d bytes) — may still reduce token count",
                log_prefix,
                len(compacted),
                len(content),
            )
        # Authoritative validation — the caller (_reduce_context) also
        # validates, but this is the canonical check that guarantees we
        # never return a malformed transcript from this function.
        if not validate_transcript(compacted):
            logger.warning("%s Compacted transcript failed validation", log_prefix)
            return None
        return compacted
    except Exception as e:
        logger.error(
            "%s Transcript compaction failed: %s", log_prefix, e, exc_info=True
        )
        return None


def _rechain_tail(compressed_prefix: str, tail_lines: list[str]) -> str:
    """Patch tail entries so their parentUuid chain links to the compressed prefix.

    The first tail entry's ``parentUuid`` is set to the ``uuid`` of the
    last entry in the compressed prefix.  Subsequent tail entries are
    rechained to point to their predecessor in the tail — their original
    ``parentUuid`` values may reference entries that were compressed away.
    """
    if not tail_lines:
        return ""
    # Find the last uuid in the compressed prefix
    last_prefix_uuid = ""
    for line in reversed(compressed_prefix.strip().split("\n")):
        if not line.strip():
            continue
        entry = json.loads(line, fallback=None)
        if isinstance(entry, dict) and "uuid" in entry:
            last_prefix_uuid = entry["uuid"]
            break

    result_lines: list[str] = []
    prev_uuid: str | None = None
    for i, line in enumerate(tail_lines):
        entry = json.loads(line, fallback=None)
        if not isinstance(entry, dict):
            # Safety guard: _find_last_assistant_entry already filters empty
            # lines, and well-formed JSONL always parses to dicts.  Non-dict
            # lines are passed through unchanged; prev_uuid is intentionally
            # NOT updated so the next dict entry chains to the last known uuid.
            result_lines.append(line)
            continue
        if i == 0:
            entry["parentUuid"] = last_prefix_uuid
        elif prev_uuid is not None:
            entry["parentUuid"] = prev_uuid
        prev_uuid = entry.get("uuid")
        result_lines.append(json.dumps(entry, separators=(",", ":")))
    return "\n".join(result_lines) + "\n"
