"""Unit tests for session file management."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from ..model import ChatMessage, ChatSession
from .session_file import _get_project_dir, cleanup_session_file, write_session_file

_NOW = datetime.now(UTC)


def _make_session(
    messages: list[ChatMessage], session_id: str = "test-session"
) -> ChatSession:
    return ChatSession(
        session_id=session_id,
        user_id="test-user",
        messages=messages,
        usage=[],
        started_at=_NOW,
        updated_at=_NOW,
    )


# -- write_session_file ------------------------------------------------------


def test_write_returns_none_for_short_history():
    """Sessions with < 2 prior messages shouldn't generate a file."""
    session = _make_session(
        [
            ChatMessage(role="user", content="hello"),
        ]
    )
    assert write_session_file(session) is None


def test_write_returns_none_for_single_pair():
    """A single user message (the current one) with no prior history."""
    session = _make_session(
        [
            ChatMessage(role="user", content="current message"),
        ]
    )
    assert write_session_file(session) is None


def test_write_creates_valid_jsonl(tmp_path: Path):
    """Multi-turn session should produce valid JSONL with correct structure."""
    session = _make_session(
        [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="Hi there!"),
            ChatMessage(role="user", content="how are you"),  # current message
        ],
        session_id="sess-123",
    )

    with patch(
        "backend.api.features.chat.sdk.session_file._get_project_dir",
        return_value=tmp_path,
    ):
        result = write_session_file(session)

    assert result == "sess-123"

    # Verify the file exists and is valid JSONL
    file_path = tmp_path / "sess-123.jsonl"
    assert file_path.exists()

    lines = file_path.read_text().strip().split("\n")
    # Should have 2 lines (prior messages only, not the current/last one)
    assert len(lines) == 2

    # Verify first line (user message)
    line1 = json.loads(lines[0])
    assert line1["type"] == "user"
    assert line1["message"]["role"] == "user"
    assert line1["message"]["content"] == "hello"
    assert line1["sessionId"] == "sess-123"
    assert line1["parentUuid"] is None  # First message has no parent
    assert "uuid" in line1
    assert "timestamp" in line1

    # Verify second line (assistant message)
    line2 = json.loads(lines[1])
    assert line2["type"] == "assistant"
    assert line2["message"]["role"] == "assistant"
    assert line2["message"]["content"] == [{"type": "text", "text": "Hi there!"}]
    assert line2["parentUuid"] == line1["uuid"]  # Chained to previous


def test_write_skips_tool_messages(tmp_path: Path):
    """Tool messages should be skipped in the session file."""
    session = _make_session(
        [
            ChatMessage(role="user", content="find agents"),
            ChatMessage(role="assistant", content="Let me search."),
            ChatMessage(role="tool", content="found 3", tool_call_id="tc1"),
            ChatMessage(role="assistant", content="I found 3 agents."),
            ChatMessage(role="user", content="run the first one"),
        ],
        session_id="sess-tools",
    )

    with patch(
        "backend.api.features.chat.sdk.session_file._get_project_dir",
        return_value=tmp_path,
    ):
        result = write_session_file(session)

    assert result == "sess-tools"
    file_path = tmp_path / "sess-tools.jsonl"
    lines = file_path.read_text().strip().split("\n")

    # Should have 3 lines: user, assistant, assistant (tool message skipped,
    # last user message excluded as current)
    assert len(lines) == 3
    types = [json.loads(line)["type"] for line in lines]
    assert types == ["user", "assistant", "assistant"]


def test_write_skips_empty_content(tmp_path: Path):
    """Messages with empty content should be skipped."""
    session = _make_session(
        [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content=""),
            ChatMessage(role="assistant", content="real response"),
            ChatMessage(role="user", content="next"),
        ],
        session_id="sess-empty",
    )

    with patch(
        "backend.api.features.chat.sdk.session_file._get_project_dir",
        return_value=tmp_path,
    ):
        result = write_session_file(session)

    assert result == "sess-empty"
    file_path = tmp_path / "sess-empty.jsonl"
    lines = file_path.read_text().strip().split("\n")
    # user + assistant (non-empty) = 2 lines
    assert len(lines) == 2


# -- cleanup_session_file ----------------------------------------------------


def test_cleanup_removes_file(tmp_path: Path):
    """cleanup_session_file should remove the session file."""
    file_path = tmp_path / "sess-cleanup.jsonl"
    file_path.write_text("{}\n")
    assert file_path.exists()

    with patch(
        "backend.api.features.chat.sdk.session_file._get_project_dir",
        return_value=tmp_path,
    ):
        cleanup_session_file("sess-cleanup")

    assert not file_path.exists()


def test_cleanup_no_error_if_missing(tmp_path: Path):
    """cleanup_session_file should not raise if file doesn't exist."""
    with patch(
        "backend.api.features.chat.sdk.session_file._get_project_dir",
        return_value=tmp_path,
    ):
        cleanup_session_file("nonexistent")  # Should not raise


# -- _get_project_dir --------------------------------------------------------


def test_get_project_dir_resolves_symlinks(tmp_path: Path):
    """_get_project_dir should resolve symlinks so the path matches the CLI."""
    # Create a symlink: tmp_path/link -> tmp_path/real
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_dir)

    with patch(
        "backend.api.features.chat.sdk.session_file._CLAUDE_PROJECTS_DIR",
        tmp_path / "projects",
    ):
        result = _get_project_dir(str(link))

    # Should resolve the symlink and encode the real path
    expected_encoded = "-" + str(real_dir).lstrip("/").replace("/", "-")
    assert result.name == expected_encoded


def test_write_uses_resolved_cwd_in_messages(tmp_path: Path):
    """The cwd field in JSONL messages should use the resolved path."""
    session = _make_session(
        [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="Hi!"),
            ChatMessage(role="user", content="current"),
        ],
        session_id="sess-cwd",
    )

    with patch(
        "backend.api.features.chat.sdk.session_file._get_project_dir",
        return_value=tmp_path,
    ):
        write_session_file(session, cwd="/tmp")

    file_path = tmp_path / "sess-cwd.jsonl"
    lines = file_path.read_text().strip().split("\n")
    for line in lines:
        obj = json.loads(line)
        # On macOS /tmp resolves to /private/tmp; on Linux stays /tmp
        resolved = str(Path("/tmp").resolve())
        assert obj["cwd"] == resolved
