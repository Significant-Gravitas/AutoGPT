"""Tests for the LocalPCShim file-ops helpers + executor-aware path resolve."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from backend.copilot.context import resolve_executor_path
from backend.copilot.sdk.e2b_file_tools import (
    _handle_edit_file,
    _handle_glob,
    _handle_grep,
    _handle_read_file,
)
from backend.copilot.sdk.file_ref import _MAX_BARE_REF_BYTES, read_file_bytes
from backend.copilot.sdk.local_pc_file_tools import (
    describe_workspace,
    is_local_pc,
    list_via_shim,
    move_via_shim,
    read_file_via_shim,
    stat_via_shim,
)
from backend.copilot.tool_image import MAX_IMAGE_BYTES
from backend.copilot.tools.local_pc_machine import MachineNotConnectedError
from backend.copilot.tools.local_pc_shim import (
    LocalPCShim,
    OpUnconfirmedError,
    ShimHello,
    ShimOverloadedError,
    ShimProtocolError,
    _CommandsProxy,
    _FilesProxy,
)


def _shim(
    allowed_root: str = "/Users/test/workspace", platform: str = "darwin"
) -> LocalPCShim:
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

    def test_windows_relative_resolves_with_windows_separators_on_linux(self):
        shim = _shim(allowed_root=r"C:\Users\test\workspace", platform="windows")
        assert (
            resolve_executor_path(r"reports\july.csv", shim)
            == r"C:\Users\test\workspace\reports\july.csv"
        )

    def test_windows_absolute_inside_root_is_case_insensitive(self):
        shim = _shim(allowed_root=r"C:\Users\Test\Workspace", platform="windows")
        assert (
            resolve_executor_path(r"c:\users\test\workspace\report.csv", shim)
            == r"c:\users\test\workspace\report.csv"
        )

    def test_windows_parent_traversal_is_rejected(self):
        shim = _shim(allowed_root=r"C:\Users\test\workspace", platform="windows")
        with pytest.raises(ValueError, match="must be within"):
            resolve_executor_path(r"C:\Users\test\workspace\..\..\secrets.txt", shim)

    def test_windows_sibling_root_is_rejected(self):
        shim = _shim(allowed_root=r"C:\workspace", platform="windows")
        with pytest.raises(ValueError, match="must be within"):
            resolve_executor_path(r"C:\workspace-old\secret.txt", shim)

    def test_windows_other_drive_is_rejected(self):
        shim = _shim(allowed_root=r"C:\workspace", platform="windows")
        with pytest.raises(ValueError, match="must be within"):
            resolve_executor_path(r"D:\workspace\secret.txt", shim)

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
        await list_via_shim(
            shim, "subdir", glob="*.csv", recursive=True, max_entries=50
        )
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


def _grep_on_shim(monkeypatch: pytest.MonkeyPatch, payload: dict) -> LocalPCShim:
    """A shim whose EXECUTE_COMMAND answers with ``payload``, wired in as the
    turn's sandbox so ``_handle_grep`` takes its Local PC branch."""
    shim = _shim(allowed_root="/Users/test/ws")
    shim.commands = _CommandsProxy(shim)
    shim._rpc = AsyncMock(return_value={"type": "COMMAND_RESULT", "payload": payload})
    monkeypatch.setattr("backend.copilot.sdk.e2b_file_tools._get_sandbox", lambda: shim)
    return shim


def _text(result: dict) -> str:
    return result["content"][0]["text"]


class TestLocalPCGrep:
    """grep exits 0 with matches, 1 with none and 2 on an error; only 1 may
    read as "No matches found."."""

    @pytest.mark.asyncio
    async def test_a_bad_pattern_is_an_error_not_an_empty_search(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _grep_on_shim(
            monkeypatch,
            {"stdout": "", "stderr": "grep: Unmatched [, [^", "exit_code": 2},
        )

        result = await _handle_grep({"pattern": "foo["})

        assert result["isError"] is True
        assert "Unmatched [" in _text(result)

    @pytest.mark.asyncio
    async def test_no_matches_is_a_normal_result(self, monkeypatch: pytest.MonkeyPatch):
        shim = _grep_on_shim(monkeypatch, {"stdout": "", "exit_code": 1})

        result = await _handle_grep({"pattern": "absent"})

        assert result["isError"] is False
        assert _text(result) == "No matches found."
        op, payload = shim._rpc.await_args.args
        assert op == "EXECUTE_COMMAND"
        assert payload["argv"][-2:] == ["absent", "/Users/test/ws"]
        assert payload["cwd"] == "/Users/test/ws"
        assert payload["timeout_seconds"] == 15

    @pytest.mark.asyncio
    async def test_truncated_matches_say_more_exist(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Past the companion's output cap the matches shown are not all of
        them, and the model must not read the rest as absent."""
        _grep_on_shim(
            monkeypatch,
            {
                "stdout": "notes.txt:3:foo\n",
                "exit_code": 0,
                "output_truncated": True,
            },
        )

        result = await _handle_grep({"pattern": "foo"})

        assert result["isError"] is False
        assert _text(result).startswith("notes.txt:3:foo")
        assert "more matches" in _text(result)

    @pytest.mark.asyncio
    async def test_matches_are_returned(self, monkeypatch: pytest.MonkeyPatch):
        _grep_on_shim(
            monkeypatch,
            {"stdout": "notes.txt:3:foo\n", "stderr": "", "exit_code": 0},
        )

        result = await _handle_grep({"pattern": "foo"})

        assert result["isError"] is False
        assert _text(result) == "notes.txt:3:foo"

    @pytest.mark.asyncio
    async def test_matches_survive_unreadable_files(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """grep exits 2 when any file was unreadable, even after matching in
        the others; those matches are real and must not be thrown away."""
        _grep_on_shim(
            monkeypatch,
            {
                "stdout": "notes.txt:3:foo\n",
                "stderr": "grep: private/key.pem: Permission denied",
                "exit_code": 2,
            },
        )

        result = await _handle_grep({"pattern": "foo"})

        assert result["isError"] is False
        assert _text(result).startswith("notes.txt:3:foo")
        assert "Permission denied" in _text(result)

    @pytest.mark.asyncio
    async def test_a_timeout_is_an_error(self, monkeypatch: pytest.MonkeyPatch):
        _grep_on_shim(
            monkeypatch,
            {"stdout": "", "stderr": "", "exit_code": -1, "timed_out": True},
        )

        result = await _handle_grep({"pattern": "foo"})

        assert result["isError"] is True
        assert "timed out" in _text(result)


class TestLocalPCGlob:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("truncated", [True, False])
    async def test_a_capped_listing_says_more_matches_exist(
        self, monkeypatch: pytest.MonkeyPatch, truncated: bool
    ):
        """The companion stops a recursive listing at its entry cap; a model
        reading only the list would take it as every match."""
        shim = _shim(allowed_root="/Users/test/ws")
        shim._rpc = AsyncMock(
            return_value={
                "type": "FILE_LIST_RESULT",
                "payload": {
                    "entries": [
                        {"path": "/Users/test/ws/a.py", "is_file": True},
                        {"path": "/Users/test/ws/pkg", "is_file": False},
                    ],
                    "truncated": truncated,
                },
            }
        )
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools._get_sandbox", lambda: shim
        )

        result = await _handle_glob({"pattern": "*.py"})

        text = _text(result)
        assert text.startswith('[\n  "/Users/test/ws/a.py"\n]')
        assert ("more matches exist" in text) is truncated

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            ShimOverloadedError("The computer is busy"),
            ShimProtocolError("The reply did not match the request"),
            OpUnconfirmedError("FILE_LIST", "w1"),
            MachineNotConnectedError("MACHINE_NOT_CONNECTED", "The computer left"),
        ],
        ids=["overloaded", "protocol", "unconfirmed", "disconnected"],
    )
    async def test_a_failed_listing_is_a_tool_error(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ):
        """A busy, broken or lost connection is a glob error like any other
        file tool's, not an exception out of the tool."""
        shim = _shim(allowed_root="/Users/test/ws")
        shim._rpc = AsyncMock(side_effect=error)
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools._get_sandbox", lambda: shim
        )

        result = await _handle_glob({"pattern": "*.py"})

        assert result["isError"] is True
        assert json.loads(_text(result)) == {
            "error": f"Glob failed: {error}",
            "type": "error",
        }


def _contents(data: bytes) -> dict:
    return {
        "type": "FILE_CONTENTS",
        "payload": {"content": base64.b64encode(data).decode("ascii")},
    }


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")
    return buffer.getvalue()


class TestLocalPCEditsTakeTurns:
    @pytest.mark.asyncio
    async def test_two_edits_to_one_file_both_land(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The CLI runs one message's tool calls in parallel, so two edits of
        one file both read it, then the second write dropped the first edit.
        Each read waits for the other edit's read, which only an edit that
        doesn't take turns reaches."""
        shim = _shim(allowed_root="/Users/test/ws")
        stored = {"/Users/test/ws/notes.txt": b"one two"}
        both_read = asyncio.Barrier(2)

        async def rpc(op: str, payload: dict, **_kwargs) -> dict:
            if op == "FILE_READ":
                data = stored[payload["path"]]
                with contextlib.suppress(TimeoutError, asyncio.BrokenBarrierError):
                    await asyncio.wait_for(both_read.wait(), 0.5)
                return _contents(data)
            stored[payload["path"]] = payload["content"].encode()
            return {"type": "ACK", "payload": {"ok": True}}

        shim._rpc = rpc
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools._get_sandbox", lambda: shim
        )

        first, second = await asyncio.gather(
            _handle_edit_file(
                {"file_path": "notes.txt", "old_string": "one", "new_string": "1"}
            ),
            _handle_edit_file(
                {"file_path": "notes.txt", "old_string": "two", "new_string": "2"}
            ),
        )

        assert first["isError"] is False and second["isError"] is False
        assert stored["/Users/test/ws/notes.txt"] == b"1 2"


class TestLocalPCEditResultSize:
    @pytest.mark.asyncio
    async def test_an_edit_over_the_limit_is_refused_before_anything_is_written(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Edit works out how large its result would be before building it, so
        a ``replace_all`` that would grow the file past the limit Read and Edit
        hold a file to is refused, and nothing is sent to the machine."""
        shim = _shim(allowed_root="/Users/test/ws")
        shim._rpc = AsyncMock(return_value=_contents(b"x-x-x-x"))
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools._get_sandbox", lambda: shim
        )
        monkeypatch.setattr("backend.copilot.sdk.e2b_file_tools._MAX_READ_BYTES", 14)

        result = await _handle_edit_file(
            {
                "file_path": "notes.txt",
                "old_string": "x",
                "new_string": "xyz",
                "replace_all": True,
            }
        )

        assert result["isError"] is True
        assert "would make notes.txt 15 bytes" in _text(result)
        assert [call.args[0] for call in shim._rpc.await_args_list] == ["FILE_READ"]


class TestLocalPCReadsThroughALimit:
    """Read, Edit and ``@@agptfile:`` references read a Local PC file through
    the caller's limit, as ``read_sandbox_file`` reads an E2B file."""

    @pytest.mark.asyncio
    async def test_read_returns_the_file(self, monkeypatch: pytest.MonkeyPatch):
        shim = _shim(allowed_root="/Users/test/ws")
        shim._rpc = AsyncMock(return_value=_contents(b"one\ntwo\n"))
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools._get_sandbox", lambda: shim
        )

        result = await _handle_read_file({"file_path": "notes.txt"})

        assert result["isError"] is False
        assert "     1\tone\n     2\ttwo\n" in _text(result)
        # The workspace file limit is over the companion's own, which a whole
        # read stops at.
        shim._rpc.assert_awaited_once_with(
            "FILE_READ", {"path": "/Users/test/ws/notes.txt", "encoding": "base64"}
        )

    @pytest.mark.asyncio
    async def test_an_image_is_read_up_to_the_image_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim(allowed_root="/Users/test/ws")
        png = _png()
        shim._rpc = AsyncMock(
            side_effect=[{"payload": {"size_bytes": len(png)}}, _contents(png)]
        )
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools._get_sandbox", lambda: shim
        )

        result = await _handle_read_file({"file_path": "shot.png"})

        assert result["isError"] is False
        assert shim._rpc.await_args_list[1].args == (
            "FILE_READ",
            {
                "path": "/Users/test/ws/shot.png",
                "encoding": "base64",
                "offset": 0,
                "length": MAX_IMAGE_BYTES + 1,
            },
        )

    @pytest.mark.asyncio
    async def test_edit_reads_the_file_and_writes_it_back(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim(allowed_root="/Users/test/ws")
        shim._rpc = AsyncMock(
            side_effect=[_contents(b"hello world"), {"type": "ACK", "payload": {}}]
        )
        monkeypatch.setattr(
            "backend.copilot.sdk.e2b_file_tools._get_sandbox", lambda: shim
        )

        result = await _handle_edit_file(
            {"file_path": "a.txt", "old_string": "world", "new_string": "there"}
        )

        assert result["isError"] is False
        read, write = shim._rpc.await_args_list
        assert read.args == (
            "FILE_READ",
            {"path": "/Users/test/ws/a.txt", "encoding": "base64"},
        )
        assert write.args[0] == "FILE_WRITE"
        assert write.args[1]["content"] == "hello there"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("platform", "root", "reference", "path"),
        [
            # A folder outside the cloud sandbox's /home/user and /tmp, which
            # the cloud resolver refused before the read reached the shim.
            ("darwin", "/Users/alice/project", "/Users/alice/project/data.csv", None),
            ("darwin", "/Users/alice/project", "data.csv", "/Users/alice/project/data.csv"),
            ("windows", "C:\\Users\\alice\\project", "data.csv", "C:\\Users\\alice\\project\\data.csv"),
            ("linux", "/home/user/ws", "/home/user/ws/data.csv", None),
        ],
        ids=["macos-absolute", "macos-relative", "windows-relative", "linux-absolute"],
    )  # fmt: skip
    async def test_a_file_reference_reads_at_most_its_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        platform: str,
        root: str,
        reference: str,
        path: str | None,
    ):
        shim = _shim(allowed_root=root, platform=platform)
        shim._rpc = AsyncMock(return_value=_contents(b"a,b\n1,2\n"))
        monkeypatch.setattr(
            "backend.copilot.sdk.file_ref.get_current_sandbox", lambda: shim
        )

        data = await read_file_bytes(reference, None, MagicMock())

        assert data == b"a,b\n1,2\n"
        shim._rpc.assert_awaited_once_with(
            "FILE_READ",
            {
                "path": path or reference,
                "encoding": "base64",
                "offset": 0,
                "length": _MAX_BARE_REF_BYTES + 1,
            },
        )

    @pytest.mark.asyncio
    async def test_a_file_reference_outside_the_folder_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim(allowed_root="/Users/alice/project", platform="darwin")
        monkeypatch.setattr(
            "backend.copilot.sdk.file_ref.get_current_sandbox", lambda: shim
        )

        with pytest.raises(ValueError, match="Path is not allowed"):
            await read_file_bytes("/Users/alice/.ssh/id_ed25519", None, MagicMock())
        shim._rpc.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_read_file_via_shim_reads_through_the_limit(self):
        shim = _shim(allowed_root="/Users/test/ws")
        shim._rpc = AsyncMock(return_value=_contents(b"12345"))

        assert await read_file_via_shim(shim, "f.txt", 4) == b"12345"
        shim._rpc.assert_awaited_once_with(
            "FILE_READ",
            {
                "path": "/Users/test/ws/f.txt",
                "encoding": "base64",
                "offset": 0,
                "length": 5,
            },
        )
