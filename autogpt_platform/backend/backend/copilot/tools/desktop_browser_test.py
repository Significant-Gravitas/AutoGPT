"""The desktop_browser tool around the driver: when it refuses, what one call
runs, and what the model gets back."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from .desktop_browser import DesktopBrowserTool
from .models import DesktopBrowserResponse, ErrorResponse

_M = "backend.copilot.tools.desktop_browser"
_D = "backend.copilot.desktop_browser"


def _session():
    return SimpleNamespace(session_id="s-1", expert_id=None)


def _box():
    return SimpleNamespace(sandbox_id="sb-1")


def _ok(result=None):
    return {"success": True, "error": None, "result": result or {}}


_OBSERVED = [
    _ok({"url": "https://a.example/"}),
    _ok({"title": "A"}),
    _ok({"snapshot": '- button "Go" [ref=e1]', "refs": {"e1": {}}}),
]


def _patched(*, screen_on=True, live=None, batches=()):
    batch = AsyncMock(side_effect=list(batches))
    patches = [
        patch(f"{_M}.get_current_sandbox", return_value=_box()),
        patch(
            "backend.copilot.computer.screen_is_on", AsyncMock(return_value=screen_on)
        ),
        patch(f"{_D}.live_scope", AsyncMock(return_value=live)),
        patch(f"{_D}.issue_scope", AsyncMock(return_value="new-scope")),
        patch(f"{_D}.ensure_browser", AsyncMock(return_value="otto-abc")),
        patch(f"{_D}.run_batch", batch),
    ]
    return patches, batch


async def _call(patches, **kwargs):
    for p in patches:
        p.start()
    try:
        return await DesktopBrowserTool()._execute("u-1", _session(), **kwargs)
    finally:
        for p in reversed(patches):
            p.stop()


@pytest.mark.asyncio
async def test_an_off_desktop_is_turned_on_by_start_desktop_not_here():
    patches, batch = _patched(screen_on=False)
    result = await _call(patches, actions=[{"action": "state"}])
    assert isinstance(result, ErrorResponse) and result.error == "desktop_off"
    assert "start_desktop" in result.message
    batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_refused_call_runs_nothing():
    patches, batch = _patched()
    result = await _call(patches, actions=[{"action": "eval", "js": "1"}])
    assert isinstance(result, ErrorResponse) and result.error == "call_refused"
    assert result.message.startswith("Nothing ran")
    batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_one_call_is_one_batch_then_a_snapshot_and_a_fresh_scope():
    patches, batch = _patched(
        live={"token": "tok", "refs": ["e1"]},
        batches=[[_ok({"clicked": "@e1"}), _ok()], _OBSERVED],
    )
    result = await _call(
        patches,
        actions=[
            {"action": "click", "ref": "@e1"},
            {"action": "press", "key": "Enter"},
        ],
        ref_scope="tok",
    )
    assert isinstance(result, DesktopBrowserResponse)
    first, second = batch.await_args_list
    assert first.args[1] == "otto-abc"
    assert first.args[2] == [["click", "@e1"], ["press", "Enter"]]
    assert first.kwargs["bail"] is True
    assert second.args[2] == [
        ["get", "url"],
        ["get", "title"],
        ["snapshot", "-i", "-c"],
    ]
    assert [r["status"] for r in result.receipts] == ["completed", "completed"]
    assert (result.url, result.title, result.ref_scope) == (
        "https://a.example/",
        "A",
        "new-scope",
    )
    assert "ref=e1" in result.snapshot
    assert result.message == "Ran 2 of 2 actions."


@pytest.mark.asyncio
async def test_a_stopped_call_says_where_it_stopped():
    patches, _ = _patched(
        live={"token": "tok", "refs": ["e1"]},
        batches=[[{"success": False, "error": "Unknown ref: e1"}], _OBSERVED],
    )
    result = await _call(
        patches,
        actions=[{"action": "click", "ref": "@e1"}, {"action": "state"}],
        ref_scope="tok",
    )
    assert isinstance(result, DesktopBrowserResponse)
    assert "Stopped at click: not_started (stale_ref)" in result.message
    assert [r["status"] for r in result.receipts] == ["not_started", "not_started"]


@pytest.mark.asyncio
async def test_a_look_is_saved_to_the_workspace():
    from .workspace_files import WorkspaceWriteResponse

    patches, _ = _patched(batches=[[_ok({"path": "/tmp/x.png"})], _OBSERVED])
    written = WorkspaceWriteResponse.model_construct(file_id="f-1", message="ok")
    patches += [
        patch(f"{_D}.read_screenshot", AsyncMock(return_value=b"\x89PNG")),
        patch(
            "backend.copilot.tools.workspace_files.WriteWorkspaceFileTool._execute",
            AsyncMock(return_value=written),
        ),
    ]
    result = await _call(patches, actions=[{"action": "look"}])
    assert isinstance(result, DesktopBrowserResponse)
    assert result.screenshot_file_id == "f-1"
    assert result.receipts[0]["result"]["file_id"] == "f-1"


@pytest.mark.asyncio
async def test_no_cloud_computer_no_browser():
    with patch(f"{_M}.get_current_sandbox", return_value=None):
        result = await DesktopBrowserTool()._execute("u-1", _session(), actions=[])
    assert isinstance(result, ErrorResponse) and result.error == "no_desktop"
