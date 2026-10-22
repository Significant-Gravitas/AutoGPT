import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from backend.copilot.sdk import service
from backend.copilot.tools import skill_matching, skill_matching_test
from backend.copilot.transcript import TranscriptDownload, cli_session_path

available_skills = skill_matching_test.available_skills


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["upload", "restore", "retry"])
@pytest.mark.parametrize("content_blocks", [False, True])
@pytest.mark.parametrize("literal_opener", [False, True])
async def test_matched_skills_do_not_replay_from_cli_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available_skills: AsyncMock,
    boundary: str,
    content_blocks: bool,
    literal_opener: bool,
) -> None:
    task = " \nFind a partner contact\n "
    notice = await skill_matching.build_matched_skills_notice(
        "user-1", skill_matching_test.expert_session(), task
    )
    assert "<matched_skills>" in notice
    query = "<available_skills>Keep this index</available_skills>\n\n" + task
    if literal_opener:
        query = "An unmatched literal <matched_skills> before the task:\n" + query
    rows = _session_rows(query, notice, content_blocks)
    raw = "\n".join(json.dumps(row) for row in rows).encode() + b"\n"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    cwd = str(tmp_path / "work")
    session_id = "11111111-1111-1111-1111-111111111111"
    path = Path(cli_session_path(cwd, session_id))
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)

    cleaned = _cross_boundary(boundary, raw, cwd, session_id, path)

    expected = _session_rows(query, "", content_blocks)
    assert [json.loads(line) for line in cleaned.splitlines()] == expected
    assert path.read_bytes() == cleaned


def _cross_boundary(
    boundary: str, raw: bytes, cwd: str, session_id: str, path: Path
) -> bytes:
    if boundary == "upload":
        uploaded = service.read_cli_session_from_disk(cwd, session_id, "[test]")
        assert uploaded is not None
        return uploaded
    if boundary == "restore":
        restored, ok = service.process_cli_restore(
            TranscriptDownload(content=raw, message_count=6, mode="sdk"),
            cwd,
            session_id,
            "[test]",
        )
        assert ok
        return restored.encode()
    service._scrub_resumed_memory_for_retry(cwd, session_id, "[test]")
    return path.read_bytes()


def _session_rows(query: str, notice: str, content_blocks: bool) -> list[dict]:
    text = query + notice
    content = (
        [
            {"type": "text", "text": text},
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.org/image.png"},
            },
        ]
        if content_blocks
        else text
    )
    return [
        {"type": "user", "uuid": "u1", "message": {"role": "user", "content": content}},
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "message": {"role": "assistant", "content": "First answer"},
        },
        {
            "type": "user",
            "uuid": "u2",
            "parentUuid": "a1",
            "message": {"role": "user", "content": content},
        },
        {
            "type": "assistant",
            "uuid": "a2",
            "parentUuid": "u2",
            "message": {"role": "assistant", "content": "Load the procedure"},
        },
        {
            "type": "user",
            "uuid": "result",
            "parentUuid": "a2",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "load",
                        "content": "Procedure body with <matched_skills>an example</matched_skills>",
                    }
                ],
            },
        },
        {
            "type": "assistant",
            "uuid": "a3",
            "parentUuid": "result",
            "message": {
                "role": "assistant",
                "content": "Keep <matched_skills>quoted text</matched_skills> here",
            },
        },
    ]
