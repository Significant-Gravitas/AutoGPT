"""Tests for the LocalPCShim file-ops helpers + executor-aware path resolve."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.copilot.context import resolve_executor_path
from backend.copilot.sdk.local_pc_file_tools import (
    describe_workspace,
    is_local_pc,
    list_via_shim,
    move_via_shim,
    stat_via_shim,
)
from backend.copilot.tools.local_pc_shim import LocalPCShim, ShimHello, _FilesProxy


def _shim(allowed_root: str = "/Users/test/workspace", platform: str = "darwin") -> LocalPCShim:
    """Build a LocalPCShim instance with _rpc stubbed."""
    shim = LocalPCShim.__new__(LocalPCShim)
    hello = ShimHello(
        machine_id="m", platform=platform, arch="arm64", allowed_root=allowed_root
    )
    shim.sandbox_id = "s1"
    shim.machine_id = hello.machine_id
    shim.platform = hello.platform
    shim.arch = hello.arch
    shim.allowed_root = hello.allowed_root
    shim.capabilities = hello.capabilities
    shim._rpc = AsyncMock(return_value={"payload": {"exists": True}})
    shim.files = _FilesProxy(shim)
    return shim


class TestResolveExecutorPath:
    """Executor-aware path resolve must use the shim's allowed_root when active."""

    def test_localpc_relative_resolves_under_allowed_root(self):
        shim = _shim(allowed_root="/Users/test/ws")
        assert resolve_executor_path("file.txt", shim) == "/Users/test/ws/file.txt"

    def test_localpc_absolute_inside_root_passes(self):
        shim = _shim(allowed_root="/Users/test/ws")
        assert (
            resolve_executor_path("/Users/test/ws/sub/file.txt", shim)
            == "/Users/test/ws/sub/file.txt"
        )

    def test_localpc_absolute_outside_root_rejected(self):
        shim = _shim(allowed_root="/Users/test/ws")
        with pytest.raises(ValueError, match="must be within"):
            resolve_executor_path("/etc/passwd", shim)

    def test_localpc_traversal_caught_after_normpath(self):
        shim = _shim(allowed_root="/Users/test/ws")
        with pytest.raises(ValueError, match="must be within"):
            resolve_executor_path("/Users/test/ws/../../../etc/passwd", shim)

    def test_localpc_sibling_root_rejected(self):
        """`/foo` must not match `/foobar` — boundary check."""
        shim = _shim(allowed_root="/Users/test/ws")
        with pytest.raises(ValueError, match="must be within"):
            resolve_executor_path("/Users/test/ws-other/file.txt", shim)

    def test_e2b_fallback_when_no_sandbox(self):
        """resolve_executor_path(path, None) preserves E2B_WORKDIR semantics."""
        assert resolve_executor_path("file.txt", None) == "/home/user/file.txt"


class TestIsLocalPc:
    def test_returns_true_for_localpcshim(self):
        assert is_local_pc(_shim()) is True

    def test_returns_false_for_plain_object(self):
        class Fake:
            pass

        assert is_local_pc(Fake()) is False

    def test_returns_false_for_none(self):
        assert is_local_pc(None) is False


class TestDescribeWorkspace:
    def test_localpc_macos(self):
        s = describe_workspace(_shim(allowed_root="/Users/a/ws", platform="darwin"))
        assert "macOS" in s
        assert "/Users/a/ws" in s

    def test_localpc_windows(self):
        s = describe_workspace(_shim(allowed_root="C:\\workspace", platform="windows"))
        assert "Windows" in s
        assert "C:\\workspace" in s

    def test_localpc_wsl2_renders_distinct_label(self):
        s = describe_workspace(_shim(allowed_root="/home/u/ws", platform="wsl2"))
        assert "WSL2" in s

    def test_non_localpc_renders_e2b_default(self):
        assert "cloud sandbox" in describe_workspace(None)


class TestStatViaShim:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_resolves_path_against_allowed_root(self):
        shim = _shim(allowed_root="/Users/test/ws")
        shim._rpc.return_value = {"payload": {"exists": True, "is_file": True}}
        await stat_via_shim(shim, "file.txt")
        # _rpc receives the resolved (absolute, under allowed_root) path
        wire_path = shim._rpc.await_args.args[1]["path"]
        assert wire_path == "/Users/test/ws/file.txt"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_jail_violation_raises_before_rpc(self):
        shim = _shim(allowed_root="/Users/test/ws")
        with pytest.raises(ValueError, match="must be within"):
            await stat_via_shim(shim, "/etc/passwd")
        shim._rpc.assert_not_called()


class TestListViaShim:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_forwards_glob_and_recursive_flags(self):
        shim = _shim()
        shim._rpc.return_value = {"payload": {"entries": [], "truncated": False}}
        await list_via_shim(shim, "subdir", glob="*.csv", recursive=True, max_entries=50)
        payload = shim._rpc.await_args.args[1]
        assert payload["glob"] == "*.csv"
        assert payload["recursive"] is True
        assert payload["max_entries"] == 50


class TestMoveViaShim:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_both_endpoints_jail_checked(self):
        shim = _shim(allowed_root="/Users/test/ws")
        shim._rpc.return_value = {"payload": {"ok": True}}
        await move_via_shim(shim, "a.csv", "sub/b.csv")
        payload = shim._rpc.await_args.args[1]
        assert payload["src"] == "/Users/test/ws/a.csv"
        assert payload["dst"] == "/Users/test/ws/sub/b.csv"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_dst_outside_root_rejected(self):
        shim = _shim(allowed_root="/Users/test/ws")
        with pytest.raises(ValueError, match="must be within"):
            await move_via_shim(shim, "a.csv", "/etc/passwd")
        shim._rpc.assert_not_called()
