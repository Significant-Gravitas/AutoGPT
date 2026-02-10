"""Session file management for Claude Code CLI --resume support.

Writes conversation history as JSONL files to the CLI's session storage
directory, enabling --resume to load full user+assistant context with
turn-level compaction support.
"""

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ..model import ChatSession

logger = logging.getLogger(__name__)

# The CLI stores sessions under ~/.claude/projects/<encoded-cwd>/<session-id>.jsonl
# The cwd path is encoded by replacing / with - and prefixing with -
_CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


def _encode_cwd(cwd: str) -> str:
    """Encode a working directory path for the CLI projects dir name."""
    return "-" + cwd.lstrip("/").replace("/", "-")


def _get_project_dir(cwd: str) -> Path:
    """Get the CLI project directory for a given working directory.

    Resolves symlinks to match the CLI's behavior (e.g. /tmp -> /private/tmp
    on macOS).
    """
    resolved = str(Path(cwd).resolve())
    return _CLAUDE_PROJECTS_DIR / _encode_cwd(resolved)


def write_session_file(
    session: ChatSession,
    cwd: str = "/tmp",
) -> str | None:
    """Write a session's conversation history as a JSONL file for --resume.

    Returns the session ID to pass to --resume, or None if there's not enough
    history to warrant a file (< 2 messages).
    """
    # Only write if there's prior conversation (at least user + assistant)
    prior = [m for m in session.messages[:-1] if m.role in ("user", "assistant")]
    if len(prior) < 2:
        return None

    session_id = session.session_id
    resolved_cwd = str(Path(cwd).resolve())
    project_dir = _get_project_dir(cwd)
    project_dir.mkdir(parents=True, exist_ok=True)

    file_path = project_dir / f"{session_id}.jsonl"
    now = datetime.now(UTC).isoformat()

    lines: list[str] = []
    prev_uuid: str | None = None

    for msg in session.messages[:-1]:
        msg_uuid = str(uuid.uuid4())

        if msg.role == "user" and msg.content:
            line = {
                "parentUuid": prev_uuid,
                "isSidechain": False,
                "userType": "external",
                "cwd": resolved_cwd,
                "sessionId": session_id,
                "type": "user",
                "message": {"role": "user", "content": msg.content},
                "uuid": msg_uuid,
                "timestamp": now,
            }
            lines.append(json.dumps(line))
            prev_uuid = msg_uuid

        elif msg.role == "assistant" and msg.content:
            line = {
                "parentUuid": prev_uuid,
                "isSidechain": False,
                "userType": "external",
                "cwd": resolved_cwd,
                "sessionId": session_id,
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": msg.content}],
                    "model": "unknown",
                },
                "uuid": msg_uuid,
                "timestamp": now,
            }
            lines.append(json.dumps(line))
            prev_uuid = msg_uuid

    if not lines:
        return None

    try:
        file_path.write_text("\n".join(lines) + "\n")
        logger.debug(f"[SESSION] Wrote {len(lines)} messages to {file_path}")
        return session_id
    except OSError as e:
        logger.warning(f"[SESSION] Failed to write session file: {e}")
        return None


def cleanup_session_file(session_id: str, cwd: str = "/tmp") -> None:
    """Remove a session file after use."""
    project_dir = _get_project_dir(cwd)
    file_path = project_dir / f"{session_id}.jsonl"
    try:
        file_path.unlink(missing_ok=True)
    except OSError:
        pass
