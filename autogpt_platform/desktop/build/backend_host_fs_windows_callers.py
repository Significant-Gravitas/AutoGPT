"""Native Windows callers; run explicitly beneath a private volume-root fixture."""

import asyncio
import contextlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.copilot.host_fs import cli_tmpdir
from backend.copilot.host_private_directory import ensure_private_directory
from backend.copilot.model import ChatMessage
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.response_model import StreamError
from backend.copilot.sdk import cli_prelaunch_turn_test as turn_tests
from backend.copilot.sdk import service
from backend.copilot.sdk.conftest import build_test_transcript
from backend.copilot.sdk.transcript_builder import TranscriptBuilder
from backend.copilot.tools import sandbox, workdir, workspace_files
from backend.copilot.transcript import TranscriptDownload


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        pytest.fail("native caller tests must mock HTTP service/provider boundaries")

    async def refuse_async(*_args, **_kwargs):
        refuse()

    monkeypatch.setattr(httpx.Client, "send", refuse)
    monkeypatch.setattr(httpx.AsyncClient, "send", refuse_async)


@pytest.fixture
def desktop_workspace(tmp_path, monkeypatch):
    assert os.name == "nt", "this suite requires native Windows"
    runtime = Path(__file__).resolve().parents[1] / "runtime"
    monkeypatch.syspath_prepend(str(runtime))
    from autogpt_desktop.layout import DataDir
    from autogpt_desktop.settings import _copilot_workspaces

    data = tmp_path / "desktop-data"
    ensure_private_directory(str(data))
    ensure_private_directory(str(data / "tmp"))
    configured = _copilot_workspaces(DataDir(data))
    prefix = os.path.realpath(configured["COPILOT_WORKSPACE_PREFIX"])
    monkeypatch.setenv("COPILOT_WORKSPACE_PREFIX", prefix)
    monkeypatch.setattr(sandbox, "WORKSPACE_PREFIX", prefix)
    monkeypatch.setattr(service, "_SDK_CWD_PREFIX", prefix)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(data / "claude"))
    monkeypatch.setattr(workdir, "get_current_sandbox", lambda: None)
    yield data, service._make_sdk_cwd("test-session-id")


def test_desktop_prefix_cli_sibling_is_private_and_pinned(desktop_workspace):
    data, cwd = desktop_workspace
    assert cwd == sandbox.make_session_path("test-session-id")
    assert Path(cwd).parent == data / "tmp"
    os.makedirs(cwd)
    sibling = Path(cli_tmpdir(cwd))
    with contextlib.ExitStack() as lifetime:
        assert service._make_cli_tmpdir(cwd, lifetime=lifetime) == str(sibling)
        assert service._make_cli_tmpdir(cwd, lifetime=lifetime) == str(sibling)
        with pytest.raises(PermissionError):
            sibling.rename(sibling.with_name(sibling.name + "-moved"))
    moved = sibling.with_name(sibling.name + "-moved")
    sibling.rename(moved)
    moved.rename(sibling)


@pytest.mark.asyncio
async def test_imported_attachment_workdir_and_tool_result_io(
    desktop_workspace, caplog
):
    _, cwd = desktop_workspace
    os.makedirs(cwd)
    with contextlib.ExitStack() as lifetime:
        service._make_cli_tmpdir(cwd, lifetime=lifetime)
        manager = SimpleNamespace(
            get_file_info=AsyncMock(
                return_value=SimpleNamespace(
                    name="attachment.txt", mime_type="text/plain", size_bytes=7
                )
            ),
            read_file_by_id=AsyncMock(return_value=b"payload"),
        )
        with patch.object(
            service, "get_workspace_manager", AsyncMock(return_value=manager)
        ):
            attached = await service._prepare_file_attachments(
                ["file-id"], "test-user", "test-session-id", cwd
            )
        assert "file_id=file-id" in attached.hint
        attachment = Path(cwd) / "attachment.txt"
        assert attachment.read_bytes() == b"payload"
        target = str(Path(cwd) / "nested" / "result.txt")
        assert (
            await workdir.save_to_workdir(target, b"tool result", "test-session-id")
            == target
        )
        assert (
            await workdir.read_workdir_bytes(target, "test-session-id")
            == b"tool result"
        )
        result = workspace_files._read_local_tool_result(
            target, 0, None, "test-session-id", sdk_cwd=cwd
        )
        assert isinstance(result, workspace_files.WorkspaceFileContentResponse)
        assert result.content == "tool result"
        assert await workdir.set_executable(
            [target], executable=True, session_id="test-session-id"
        ) == [target]
        assert "POSIX executable modes are unavailable on Windows" in caplog.text
        assert Path(target).read_bytes() == b"tool result"
        assert await workdir.remove_from_workdir([target], "test-session-id") == []
        assert not Path(target).exists()


@pytest.mark.asyncio
async def test_imported_transcript_write_read_and_resume(desktop_workspace):
    _, cwd = desktop_workspace
    os.makedirs(cwd)
    transcript = build_test_transcript([("user", "question"), ("assistant", "answer")])
    with contextlib.ExitStack() as lifetime:
        service._make_cli_tmpdir(cwd, lifetime=lifetime)
        assert service._write_cli_session_to_disk(
            transcript.encode(), cwd, "test-session-id", "native-test"
        )
        restored = service.read_cli_session_from_disk(
            cwd, "test-session-id", "native-test"
        )
        assert restored is not None and b"answer" in restored
        session = turn_tests._turns._make_session()
        session.messages[:0] = [
            ChatMessage(role="user", content="question"),
            ChatMessage(role="assistant", content="answer"),
        ]
        builder = TranscriptBuilder()
        with (
            patch.object(service.config, "claude_agent_use_resume", True),
            patch.object(
                service,
                "download_transcript",
                AsyncMock(
                    return_value=TranscriptDownload(
                        content=transcript.encode(), message_count=2, mode="sdk"
                    )
                ),
            ),
        ):
            resume = await service._restore_cli_session_for_turn(
                "test-user", "test-session-id", session, cwd, builder, "native-test"
            )
        assert resume.use_resume and resume.resume_file == "test-session-id"
        assert builder.entry_count == 2
        assert Path(service.cli_session_path(cwd, "test-session-id")).is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["success", "early", "failure", "cancel"])
async def test_real_turn_holds_cli_directory_until_transport_cleanup(
    desktop_workspace, ending
):
    _, cwd = desktop_workspace
    real_makedirs = os.makedirs
    real_make_cwd = service._make_sdk_cwd
    session = turn_tests._turns._make_session()
    message = "hello" if ending == "success" else "   "
    session.messages[-1].content = message
    turn_tests._FakeTransport.spawned = []
    started = asyncio.Event()
    closing = []
    held = PendingMessage(content="<held_call_result>done</held_call_result>")

    async def resolve(_user_id, _session, deliver, **_kwargs):
        if ending in ("failure", "cancel"):
            deliver(held)
        return
        yield

    async def persist(*_args, **_kwargs):
        started.set()
        if ending == "failure":
            raise RuntimeError("owned persistence failure")
        await asyncio.Event().wait()

    async def close(transport):
        sibling = Path(cli_tmpdir(cwd))
        with pytest.raises(PermissionError):
            sibling.rename(sibling.with_name(sibling.name + "-premature"))
        closing.append(sibling)
        transport.closed = True

    def client_factory(*_args, **kwargs):
        assert ending == "success", "empty prompt must never start a provider query"
        client = turn_tests._turns._make_client_mock(
            result_message=turn_tests._turns._make_result_message()
        )

        async def exit_client(*_exc):
            await close(kwargs["transport"])

        client.__aexit__.side_effect = exit_client
        return client

    events = []

    async def consume():
        async for event in service.stream_chat_completion_sdk(
            session_id="test-session-id",
            message=message,
            is_user_message=True,
            user_id="test-user",
            session=session,
        ):
            events.append(event)

    with (
        turn_tests._patched_turn(
            client_factory, session, resolve=resolve, persist=persist
        ),
        patch.object(os, "makedirs", real_makedirs),
        patch.object(service, "_make_sdk_cwd", real_make_cwd),
        patch.object(turn_tests._FakeTransport, "close", close),
        patch.object(
            service, "first_message_expert_context", AsyncMock(return_value="")
        ),
        patch.object(service, "inject_skill_invocation", AsyncMock(return_value=None)),
    ):
        task = asyncio.create_task(consume())
        try:
            if ending == "cancel":
                await asyncio.wait_for(started.wait(), 10)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 10)
            else:
                await asyncio.wait_for(task, 10)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    if ending == "success":
        assert not any(isinstance(event, StreamError) for event in events)
    else:
        assert any(
            isinstance(event, StreamError) and event.code == "empty_prompt"
            for event in events
        )
    assert len(closing) == 1
    assert turn_tests._FakeTransport.spawned[0].closed
    sibling = Path(cli_tmpdir(cwd))
    if sibling.exists():
        moved = sibling.with_name(sibling.name + "-released")
        sibling.rename(moved)
        moved.rename(sibling)
