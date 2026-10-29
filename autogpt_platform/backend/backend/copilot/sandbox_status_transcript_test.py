"""A replacement notice reaches the current query, never a resumed turn."""

import json
from pathlib import Path

import pytest

from backend.copilot import transcript
from backend.copilot.sandbox_status import SANDBOX_REPLACED_NOTICE
from backend.copilot.sdk import service


@pytest.mark.parametrize(
    "content, expected",
    [
        (SANDBOX_REPLACED_NOTICE + "continue", "continue"),
        (
            [{"type": "text", "text": SANDBOX_REPLACED_NOTICE + "continue"}],
            [{"type": "text", "text": "continue"}],
        ),
        (
            [
                {"type": "text", "text": SANDBOX_REPLACED_NOTICE},
                {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
            ],
            [{"type": "image", "source": {"type": "base64", "data": "AAAA"}}],
        ),
    ],
)
def test_saved_queries_lose_the_notice_but_keep_user_content(content, expected):
    entry = {
        "type": "user",
        "uuid": "u1",
        "message": {"role": "user", "content": content},
    }
    raw = json.dumps(entry) + "\n"

    cleaned = transcript.strip_for_upload(raw)

    assert "sandbox_status" not in cleaned
    assert json.loads(cleaned)["message"]["content"] == expected
    assert transcript.strip_for_upload(cleaned) == cleaned


@pytest.mark.parametrize(
    "role, content",
    [("user", "explain sandbox_status"), ("assistant", SANDBOX_REPLACED_NOTICE)],
)
def test_other_entries_keep_their_original_bytes(role, content):
    raw = (
        json.dumps({"type": role, "message": {"role": role, "content": content}}) + "\n"
    )
    assert transcript.strip_for_upload(raw) == raw


def test_cli_upload_and_restore_remove_the_notice(tmp_path, monkeypatch):
    projects = str(tmp_path / "projects")
    monkeypatch.setattr(transcript, "projects_base", lambda: projects)
    monkeypatch.setattr(service, "projects_base", lambda: projects)
    cwd = str(tmp_path / "workspace")
    session_id = "12345678-0000-0000-0000-aabbccdd0001"
    session_file = Path(transcript.cli_session_path(cwd, session_id))
    session_file.parent.mkdir(parents=True)
    raw = (
        json.dumps(
            {
                "type": "user",
                "uuid": "u1",
                "message": {
                    "role": "user",
                    "content": SANDBOX_REPLACED_NOTICE + "continue",
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "uuid": "a1",
                "parentUuid": "u1",
                "message": {"role": "assistant", "content": "done"},
            }
        )
        + "\n"
    )
    session_file.write_text(raw)

    uploaded = service.read_cli_session_from_disk(cwd, session_id, "[Test]")
    assert uploaded is not None and b"sandbox_status" not in uploaded
    assert b"continue" in uploaded
    assert session_file.read_bytes() == uploaded

    restored, ok = service.process_cli_restore(
        transcript.TranscriptDownload(content=raw, message_count=2),
        cwd,
        session_id,
        "[Test]",
    )
    assert ok
    assert "sandbox_status" not in restored
    assert "continue" in restored
    assert session_file.read_text() == restored
