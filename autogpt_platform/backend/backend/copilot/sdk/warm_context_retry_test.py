import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from backend.copilot.sdk import service
from backend.copilot.transcript import cli_session_path

from .retry_scenarios_test import TestFollowUpWarmContextCallSite as _Turn

_SVC = "backend.copilot.sdk.service"
_OLD = "<temporal_context>forgotten before retry</temporal_context>"
_FRESH = "<temporal_context>current memory</temporal_context>"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "handled"])
@pytest.mark.parametrize("scrub_write_fails", [False, True])
async def test_retry_scrubs_the_file_before_the_next_cli_resumes(
    tmp_path, monkeypatch, failure, scrub_write_fails
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    path = Path(cli_session_path("/tmp/test-sdk-cwd", "test-session-id"))
    path.parent.mkdir(parents=True)
    prior = b'{"type":"assistant","message":{"role":"assistant","content":"prior answer"}}\n'
    path.write_bytes(prior)
    restored: list[bytes] = []
    queries: list[str] = []

    def launch(*args, **kwargs):
        assert kwargs["options"].resume == "test-session-id"
        restored.append(path.read_bytes())
        attempt = len(restored)

        async def query(prompt, session_id=None):
            queries.append(prompt)
            entry = {"type": "user", "message": {"role": "user", "content": prompt}}
            with path.open("ab") as stream:
                stream.write(json.dumps(entry).encode() + b"\n")
            if attempt == 1 and failure == "exception":
                raise ConnectionError("ECONNRESET")

        async def receive():
            if attempt == 1 and failure == "handled":
                yield AssistantMessage(
                    content=[TextBlock(text="API Error: 529 Overloaded")],
                    model="anthropic/claude-sonnet-4-6",
                    error="server_error",
                )
            yield ResultMessage(
                subtype="success",
                result="done",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="test-session-id",
                total_cost_usd=0.0,
            )

        client = MagicMock(query=query, receive_response=receive)
        client._transport.write = AsyncMock()
        manager = AsyncMock()
        manager.__aenter__.return_value = client
        manager.__aexit__.return_value = None
        return manager

    with (
        pytest.raises(RuntimeError, match="Cannot save the session for retry")
        if scrub_write_fails
        else nullcontext()
    ):
        await _Turn()._run(
            _Turn()._session("restart the executor"),
            AsyncMock(side_effect=[_OLD, _FRESH]),
            launch,
            extra=[
                (
                    f"{_SVC}._restore_cli_session_for_turn",
                    dict(
                        new=AsyncMock(
                            return_value=service._RestoreResult(
                                transcript_content=prior.decode(),
                                transcript_covers_prefix=True,
                                use_resume=True,
                                resume_file="test-session-id",
                                transcript_msg_count=2,
                                context_messages=[],
                            )
                        )
                    ),
                ),
                (f"{_SVC}.strip_for_upload", dict(new=lambda text: text)),
                (f"{_SVC}._compute_transient_backoff", dict(return_value=0)),
                (f"{_SVC}.persist_and_record_usage", dict(new=AsyncMock())),
                *(
                    [(f"{_SVC}._write_cli_session_to_disk", dict(return_value=False))]
                    if scrub_write_fails
                    else []
                ),
            ],
        )

    if scrub_write_fails:
        assert len(restored) == len(queries) == 1
        return
    assert len(restored) == len(queries) == 2
    assert "forgotten before retry" in queries[0]
    assert "current memory" in queries[1]
    assert "forgotten before retry" not in queries[1]
    assert b"forgotten before retry" not in restored[1]
    assert b"restart the executor" in restored[1]
    assert restored[1].startswith(prior)


def test_retry_leaves_an_unmarked_file_byte_for_byte(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    path = Path(cli_session_path("/tmp/test-sdk-cwd", "active-resume"))
    path.parent.mkdir(parents=True)
    content = b'{"type":"user", "message":{"role":"user","content":"unchanged"}}\r\n'
    path.write_bytes(content)

    with patch.object(service, "_write_cli_session_to_disk") as write:
        service._scrub_resumed_memory_for_retry(
            "/tmp/test-sdk-cwd", "active-resume", "[test]"
        )

    write.assert_not_called()
    assert path.read_bytes() == content


def test_retry_stops_when_the_resume_file_cannot_be_read(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    with (
        patch.object(Path, "read_bytes", side_effect=PermissionError),
        pytest.raises(RuntimeError, match="Cannot read the session for retry"),
    ):
        service._scrub_resumed_memory_for_retry(
            "/tmp/test-sdk-cwd", "active-resume", "[test]"
        )
