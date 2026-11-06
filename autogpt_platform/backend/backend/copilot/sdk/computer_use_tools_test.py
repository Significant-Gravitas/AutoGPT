"""Handler-level coverage for Otto's Local PC computer-use MCP tools."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import backend.copilot.sdk.computer_use_tools as tools
from backend.copilot.tools.local_pc_shim import (
    ComputerUseDriver,
    LocalPCShim,
    ShimComputerUseError,
)

_TOOLS = ("click", "get_window_state", "list_windows", "type_text")
_CATALOG = {
    "driver": {"name": "cua-driver", "version": "0.38.0", "runtime": "direct"},
    "instructions": "cua-driver: background GUI automation.",
    "tools": [
        {
            "name": "get_window_state",
            "description": "Read a window.\n\nLong details the summary leaves out.",
            "input_schema": {"type": "object", "properties": {"pid": {}}},
            "annotations": {"readOnlyHint": True},
        },
        {
            "name": "click",
            "description": "Click an element. " + "x" * 400,
            "input_schema": {"type": "object", "properties": {"element_token": {}}},
            "annotations": None,
        },
    ],
}


@pytest.fixture(autouse=True)
def _user_in_the_computer_use_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Its rollout flags are off by default; a user in the beta has them on."""
    monkeypatch.setattr(tools, "is_computer_use_enabled", AsyncMock(return_value=True))


def _shim(*, capabilities: tuple[str, ...] = ("computer_use",)) -> LocalPCShim:
    shim = LocalPCShim.__new__(LocalPCShim)
    shim.sandbox_id = "session-1"
    shim.platform = "linux"
    shim.machine_id = "machine-1"
    shim.allowed_root = "/home/test/workspace"
    shim.capabilities = list(capabilities)
    shim.capability_set = frozenset(capabilities)
    shim.computer_use_driver = ComputerUseDriver(
        name="cua-driver", version="0.38.0", runtime="direct"
    )
    shim.computer_use_tools = list(_TOOLS)
    shim._connection_generation = 1
    shim.computer = MagicMock()
    shim.computer.tools = AsyncMock(return_value=_CATALOG)
    shim.computer.call = AsyncMock(
        return_value={
            "tool": "click",
            "is_error": False,
            "content": [{"type": "text", "text": "Clicked."}],
            "structured_content": {"effect": "confirmed"},
        }
    )
    return shim


def _install(monkeypatch: pytest.MonkeyPatch, shim: LocalPCShim | None) -> None:
    monkeypatch.setattr(tools, "get_current_sandbox", lambda: shim)
    monkeypatch.setattr(
        tools,
        "get_execution_context",
        lambda: ("user-1", SimpleNamespace(session_id="session-1")),
    )
    monkeypatch.setattr(tools, "is_computer_use_approved", AsyncMock(return_value=True))


def _payload(result: dict) -> dict:
    return json.loads(result["content"][0]["text"])


class TestGating:
    @pytest.mark.asyncio
    async def test_no_executor_fails_closed(self, monkeypatch: pytest.MonkeyPatch):
        _install(monkeypatch, None)

        result = await tools._h_computer({"tool": "click"})

        assert result["isError"] is True
        assert _payload(result)["code"] == "NO_LOCAL_PC_EXECUTOR"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("handler", ["_h_computer", "_h_computer_tools"])
    async def test_a_user_outside_the_computer_use_rollout_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, handler: str
    ):
        """A call held for approval replays through the handler on a later
        turn, maybe after the flag turned computer use off for the user."""
        shim = _shim()
        _install(monkeypatch, shim)
        gate = AsyncMock(return_value=False)
        monkeypatch.setattr(tools, "is_computer_use_enabled", gate)

        result = await getattr(tools, handler)({"tool": "click"})

        assert _payload(result)["code"] == "COMPUTER_USE_DISABLED"
        gate.assert_awaited_once_with("user-1")
        shim.computer.call.assert_not_awaited()
        shim.computer.tools.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_capability_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _install(monkeypatch, _shim(capabilities=("files",)))

        result = await tools._h_computer_tools({})

        assert result["isError"] is True
        assert _payload(result)["code"] == "CAPABILITY_NOT_GRANTED"

    @pytest.mark.asyncio
    async def test_a_shim_without_a_driver_offers_no_computer_use(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim()
        shim.computer_use_driver = None
        _install(monkeypatch, shim)

        result = await tools._h_computer({"tool": "click"})

        assert _payload(result)["code"] == "CAPABILITY_NOT_GRANTED"
        shim.computer.call.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_consent_is_checked_against_the_machine_driver_and_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim()
        _install(monkeypatch, shim)
        consent = AsyncMock(return_value=False)
        monkeypatch.setattr(tools, "is_computer_use_approved", consent)

        result = await tools._h_computer({"tool": "click", "arguments": {}})

        assert result["isError"] is True
        assert _payload(result)["code"] == "COMPUTER_USE_CONSENT_REQUIRED"
        consent.assert_awaited_once_with(
            "session-1",
            "user-1",
            machine_id="machine-1",
            driver="cua-driver 0.38.0",
            tools=_TOOLS,
        )
        shim.computer.call.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reconnect_during_consent_check_blocks_dispatch(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim()
        _install(monkeypatch, shim)
        consent_started = asyncio.Event()
        release_consent = asyncio.Event()

        async def delayed_approval(*_args, **_kwargs):
            consent_started.set()
            await release_consent.wait()
            return True

        monkeypatch.setattr(tools, "is_computer_use_approved", delayed_approval)
        task = asyncio.create_task(tools._h_computer({"tool": "click"}))
        await consent_started.wait()
        shim._connection_generation += 1
        shim.computer_use_tools = [*_TOOLS, "launch_app"]
        release_consent.set()

        result = await task

        assert result["isError"] is True
        assert _payload(result)["code"] == "COMPUTER_USE_CONNECTION_CHANGED"
        shim.computer.call.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_tool_the_shim_does_not_offer_never_reaches_it(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim()
        _install(monkeypatch, shim)

        result = await tools._h_computer({"tool": "kill_app", "arguments": {"pid": 1}})

        assert _payload(result)["code"] == "FEATURE_NOT_SUPPORTED"
        assert _payload(result)["details"] == {"tool": "kill_app"}
        shim.computer.call.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "args",
        [
            {},
            {"tool": "Click"},
            {"tool": "click", "arguments": [1, 2]},
            {"tool": "click", "timeout_seconds": 0},
            {"tool": "click", "timeout_seconds": 601},
            {"tool": "click", "timeout_seconds": "10"},
        ],
    )
    async def test_malformed_calls_are_refused(
        self, monkeypatch: pytest.MonkeyPatch, args: dict
    ):
        shim = _shim()
        _install(monkeypatch, shim)

        result = await tools._h_computer(args)

        assert _payload(result)["code"] == "INVALID_ARGUMENT"
        shim.computer.call.assert_not_awaited()


class TestToolList:
    @pytest.mark.asyncio
    async def test_lists_guidance_and_summaries(self, monkeypatch: pytest.MonkeyPatch):
        shim = _shim()
        _install(monkeypatch, shim)

        result = await tools._h_computer_tools({})

        view = _payload(result)
        assert view["instructions"] == "cua-driver: background GUI automation."
        assert view["driver"]["name"] == "cua-driver"
        assert view["tools"][0] == {
            "name": "get_window_state",
            "summary": "Read a window.",
        }
        assert len(view["tools"][1]["summary"]) == 280
        assert "schemas" not in view

    @pytest.mark.asyncio
    async def test_returns_the_schemas_asked_for(self, monkeypatch: pytest.MonkeyPatch):
        shim = _shim()
        _install(monkeypatch, shim)

        result = await tools._h_computer_tools({"tools": ["click", "page"]})

        view = _payload(result)
        assert view["schemas"]["click"]["input_schema"] == {
            "type": "object",
            "properties": {"element_token": {}},
        }
        assert view["unknown_tools"] == ["page"]

    @pytest.mark.asyncio
    async def test_a_shim_error_is_structured(self, monkeypatch: pytest.MonkeyPatch):
        shim = _shim()
        shim.computer.tools = AsyncMock(
            side_effect=ShimComputerUseError(
                "DEPENDENCY_MISSING", "cua-driver is not installed", {"x": 1}
            )
        )
        _install(monkeypatch, shim)

        result = await tools._h_computer_tools({})

        assert _payload(result) == {
            "code": "DEPENDENCY_MISSING",
            "error": "cua-driver is not installed",
            "details": {"x": 1},
        }


class TestCall:
    @pytest.mark.asyncio
    async def test_relays_the_call_and_the_drivers_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim()
        _install(monkeypatch, shim)

        result = await tools._h_computer(
            {
                "tool": "click",
                "arguments": {"element_token": "s1:4"},
                "timeout_seconds": 30,
            }
        )

        call = shim.computer.call.await_args
        assert call.args == ("click", {"element_token": "s1:4"})
        assert call.kwargs["timeout_seconds"] == 30
        assert result == {
            "content": [
                {"type": "text", "text": "Clicked."},
                {"type": "text", "text": json.dumps({"effect": "confirmed"})},
            ],
            "isError": False,
        }

    @pytest.mark.asyncio
    async def test_images_reach_the_model(self, monkeypatch: pytest.MonkeyPatch):
        shim = _shim()
        tree = "- frame\n" + '  - [1] button "Save"\n' * 20
        shim.computer.call = AsyncMock(
            return_value={
                "tool": "get_window_state",
                "is_error": False,
                "content": [
                    {"type": "text", "text": tree},
                    {"type": "image", "data": "aW1n", "mime_type": "image/png"},
                ],
                "structured_content": {"tree_markdown": tree, "snapshot_id": "s1"},
            }
        )
        _install(monkeypatch, shim)

        result = await tools._h_computer({"tool": "get_window_state"})

        content = result["content"]
        assert content[1] == {"type": "image", "data": "aW1n", "mimeType": "image/png"}
        # The tree is in the text already; it isn't sent twice.
        assert json.loads(content[2]["text"]) == {
            "tree_markdown": "(shown in the text above)",
            "snapshot_id": "s1",
        }

    @pytest.mark.asyncio
    async def test_the_drivers_error_is_an_error_result(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim()
        shim.computer.call = AsyncMock(
            return_value={
                "tool": "click",
                "is_error": True,
                "content": [{"type": "text", "text": "The element token is stale."}],
                "structured_content": None,
            }
        )
        _install(monkeypatch, shim)

        result = await tools._h_computer({"tool": "click"})

        assert result["isError"] is True
        assert result["content"] == [
            {"type": "text", "text": "The element token is stale."}
        ]

    @pytest.mark.asyncio
    async def test_a_shim_refusal_is_structured(self, monkeypatch: pytest.MonkeyPatch):
        shim = _shim()
        shim.computer.call = AsyncMock(
            side_effect=ShimComputerUseError(
                "CLIPBOARD_CONCEALED",
                "A password manager marked the clipboard private.",
                {"reason": "concealed_type"},
            )
        )
        _install(monkeypatch, shim)

        result = await tools._h_computer({"tool": "click"})

        assert _payload(result)["code"] == "CLIPBOARD_CONCEALED"

    def test_only_the_call_tool_is_registered_as_mutating(self):
        assert tools.LOCAL_PC_COMPUTER_TOOL_NAMES == [
            "local_pc_computer_tools",
            "local_pc_computer",
        ]
