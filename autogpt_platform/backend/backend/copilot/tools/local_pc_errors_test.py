"""Tests for the LocalPC shim error code → English translator.

Each code in the wire ERROR envelope should map to an actionable recovery
hint for the LLM. A renamed translation should break an explicit test,
not silently change agent behavior.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from .local_pc_errors import translate_shim_error
from .local_pc_shim import LocalPCShim, ShimHello, _CommandsProxy, _FilesProxy


def _make_shim(
    *,
    allowed_root: str = "/Users/test/ws",
    platform: str = "darwin",
    arch: str = "arm64",
) -> LocalPCShim:
    shim = LocalPCShim.__new__(LocalPCShim)
    shim.sandbox_id = "s1"
    shim.allowed_root = allowed_root
    shim.platform = platform
    shim.arch = arch
    shim.machine_id = "m1"
    shim.capabilities = []
    return shim


# Sentinel substrings that prove each code rendered the right message.
# Pairs (code, expected_substring) — fragile-on-purpose so a translation
# rename trips a test instead of silently shifting agent recovery hints.
_EXPECTED_SUBSTRINGS: dict[str, list[str]] = {
    "PATH_OUTSIDE_ALLOWED_ROOT": ["outside", "workspace", "/Users/test/ws"],
    "PATH_RESERVED_NAME": ["Windows reserved name", "CON"],
    "PATH_INVALID_CHARS": ["illegal", "Sanitize"],
    "PATH_NOT_FOUND": ["doesn't exist", "list"],
    "PATH_NOT_EMPTY": ["not empty", "recursive=true"],
    "PATH_EXISTS": ["already exists", "overwrite=true"],
    "COMMAND_TIMEOUT": ["timeout", "killed"],
    "SHELL_NOT_AVAILABLE": ["isn't installed", "auto"],
    "PERMISSION_PENDING": ["macOS", "System Settings", "autogpt-shim doctor"],
    "CLIPBOARD_CONCEALED": ["password manager", "private"],
    "FEATURE_NOT_SUPPORTED": ["isn't available", "local_pc_computer_tools"],
    "CAPABILITY_NOT_GRANTED": ["didn't grant", "autogpt-shim auth"],
    "SHIM_OVERLOADED": ["concurrency cap", "Retry"],
    "AUTH_FAILED": ["OAuth", "autogpt-shim auth"],
    "UNSUPPORTED_ARCH": ["architecture", "x86_64", "arm64"],
    "WRITE_UNCONFIRMED": ["FILE_WRITE", "FILE_STAT"],
    "FILE_TOO_LARGE": ["exceeded", "Chunk", "offset+length"],
    "DEPENDENCY_MISSING": ["pipx install", "autogpt-local-executor"],
    "RECORDING_NOT_FOUND": ["No recording", "list_recordings"],
    "RECORDING_CHANNEL_UNAVAILABLE": [
        "capture channel isn't available",
        "screenshot+action floor",
        "visual replay",
    ],
    "RECORDING_ALREADY_ACTIVE": [
        "already in progress",
        "AutoGPT recording control",
    ],
    "CONSENT_REQUIRED": ["consent token", "OS-native", "platform cannot"],
    "INTERPRETATION_UNAVAILABLE": [
        "local model",
        "re-record",
        "screenshots_to_cloud",
    ],
    "INTERNAL_ERROR": ["unexpected internal error", "autogpt-shim audit tail"],
}


class TestTranslationTable:
    """Each documented code must produce its canonical recovery hint."""

    @pytest.mark.parametrize("code", list(_EXPECTED_SUBSTRINGS.keys()))
    def test_renders_expected_substrings(self, code: str):
        shim = _make_shim()
        out = translate_shim_error(code, "raw wire message", {}, shim)
        for needle in _EXPECTED_SUBSTRINGS[code]:
            assert needle in out, f"{code}: expected '{needle}' in '{out}'"


class TestPlaceholderSubstitution:
    """Translator must actually weave in shim metadata / details when present."""

    def test_path_outside_allowed_root_includes_allowed_root_literal(self):
        shim = _make_shim(allowed_root="/Users/test/ws")
        out = translate_shim_error(
            "PATH_OUTSIDE_ALLOWED_ROOT",
            "Path /etc/passwd is outside allowed root /Users/test/ws",
            {"path": "/etc/passwd"},
            shim,
        )
        assert "/Users/test/ws" in out
        assert "/etc/passwd" in out

    def test_path_outside_allowed_root_falls_back_when_no_shim(self):
        out = translate_shim_error(
            "PATH_OUTSIDE_ALLOWED_ROOT",
            "Path /etc/passwd is outside allowed root",
            {"path": "/etc/passwd"},
            None,
        )
        # Should not crash, should still mention the path.
        assert "/etc/passwd" in out
        assert "workspace" in out

    def test_path_invalid_chars_mentions_platform(self):
        shim = _make_shim(platform="windows")
        out = translate_shim_error("PATH_INVALID_CHARS", "bad chars", {}, shim)
        assert "windows" in out

    def test_path_not_found_uses_details_path(self):
        out = translate_shim_error(
            "PATH_NOT_FOUND",
            "not found",
            {"path": "/Users/test/ws/missing.txt"},
            _make_shim(),
        )
        assert "/Users/test/ws/missing.txt" in out

    def test_path_not_empty_uses_details_path(self):
        out = translate_shim_error(
            "PATH_NOT_EMPTY",
            "dir not empty",
            {"path": "/Users/test/ws/cache"},
            _make_shim(),
        )
        assert "/Users/test/ws/cache" in out

    def test_path_exists_uses_details_dst(self):
        out = translate_shim_error(
            "PATH_EXISTS",
            "dst exists",
            {"dst": "/Users/test/ws/out.json"},
            _make_shim(),
        )
        assert "/Users/test/ws/out.json" in out

    def test_command_timeout_includes_seconds(self):
        out = translate_shim_error(
            "COMMAND_TIMEOUT", "timed out", {"timeout_seconds": 30}, _make_shim()
        )
        assert "30s" in out

    def test_shell_not_available_includes_requested_shell(self):
        shim = _make_shim(platform="windows")
        out = translate_shim_error(
            "SHELL_NOT_AVAILABLE",
            "bash not found",
            {"requested_shell": "bash"},
            shim,
        )
        assert "bash" in out
        assert "windows" in out

    def test_permission_pending_names_permission(self):
        out = translate_shim_error(
            "PERMISSION_PENDING",
            "denied",
            {"permission": "Screen Recording"},
            _make_shim(),
        )
        assert "Screen Recording" in out

    def test_clipboard_concealed_names_the_marker(self):
        out = translate_shim_error(
            "CLIPBOARD_CONCEALED",
            "concealed",
            {"reason": "concealed_type", "marker": "org.nspasteboard.ConcealedType"},
            _make_shim(),
        )
        assert "org.nspasteboard.ConcealedType" in out

    @pytest.mark.parametrize(
        ("reason", "needles"),
        [
            ("writeback_only", ["clipboard_write", "30 seconds"]),
            ("writeback_overwritten", ["replaced it since"]),
            ("changed", ["Read it again"]),
        ],
    )
    def test_clipboard_concealed_explains_the_reason(self, reason, needles):
        out = translate_shim_error(
            "CLIPBOARD_CONCEALED", "concealed", {"reason": reason}, _make_shim()
        )
        for needle in needles:
            assert needle in out

    def test_feature_not_supported_includes_tool_and_platform(self):
        shim = _make_shim(platform="linux")
        out = translate_shim_error(
            "FEATURE_NOT_SUPPORTED",
            "kill_app is not available on this machine.",
            {"tool": "kill_app"},
            shim,
        )
        assert "`kill_app`" in out
        assert "linux" in out
        assert "kill_app is not available on this machine." in out

    def test_a_computer_tool_timeout_names_the_tool(self):
        out = translate_shim_error(
            "COMMAND_TIMEOUT",
            "drag did not finish in time",
            {"tool": "drag", "timeout_seconds": 120},
            _make_shim(),
        )
        assert "`drag`" in out
        assert "120s" in out
        assert "stays done" in out

    def test_capability_not_granted_includes_capability_name(self):
        out = translate_shim_error(
            "CAPABILITY_NOT_GRANTED",
            "missing scope",
            {"capability": "clipboard"},
            _make_shim(),
        )
        assert "clipboard" in out

    def test_shim_overloaded_includes_max_concurrent(self):
        out = translate_shim_error(
            "SHIM_OVERLOADED", "busy", {"max_concurrent": 4}, _make_shim()
        )
        assert "4" in out

    def test_unsupported_arch_uses_shim_arch(self):
        shim = _make_shim(arch="ppc64")
        out = translate_shim_error("UNSUPPORTED_ARCH", "bad arch", {}, shim)
        assert "ppc64" in out

    def test_internal_error_passes_through_raw_message(self):
        out = translate_shim_error(
            "INTERNAL_ERROR", "segfault in adapter", {}, _make_shim()
        )
        assert "segfault in adapter" in out

    def test_file_too_large_includes_max_size_and_op(self):
        out = translate_shim_error(
            "FILE_TOO_LARGE",
            "too big",
            {
                "max_file_size_bytes": 10485760,
                "attempted_bytes": 52428800,
                "op": "FILE_WRITE",
            },
            _make_shim(),
        )
        assert "10485760 bytes" in out
        assert "52428800 bytes" in out
        assert "FILE_WRITE" in out
        assert "offset+length" in out

    def test_file_too_large_fallback_when_details_missing(self):
        out = translate_shim_error("FILE_TOO_LARGE", "too big", {}, _make_shim())
        # Generic fallback wording must still be actionable.
        assert "Chunk the read/write" in out
        assert "shim's configured limit" in out

    def test_file_too_large_op_only_uses_op_in_chunk_clause(self):
        out = translate_shim_error(
            "FILE_TOO_LARGE",
            "too big",
            {"op": "FILE_READ"},
            _make_shim(),
        )
        assert "Chunk the FILE_READ" in out

    def test_dependency_missing_includes_dep_and_extra(self):
        out = translate_shim_error(
            "DEPENDENCY_MISSING",
            "Pillow not installed",
            {"dep": "pyserial", "extra": "hardware", "op": "SERIAL_OPEN"},
            _make_shim(),
        )
        assert "pyserial" in out
        assert "pipx install autogpt-local-executor[hardware]" in out
        assert "SERIAL_OPEN" in out

    def test_dependency_missing_for_cua_driver_points_at_the_extra(self):
        out = translate_shim_error(
            "DEPENDENCY_MISSING",
            "cua-driver is not installed.",
            {"dependency": "cua-driver"},
            _make_shim(),
        )
        assert "autogpt-local-executor[computer-use]" in out
        assert "autogpt-shim doctor" in out

    def test_dependency_missing_falls_back_to_passthrough_without_dep(self):
        out = translate_shim_error(
            "DEPENDENCY_MISSING",
            "missing some package",
            {},
            _make_shim(),
        )
        # No dep → passthrough message, but still mentions the install path.
        assert "missing some package" in out
        assert "pipx install autogpt-local-executor" in out

    def test_dependency_missing_without_extra_uses_placeholder(self):
        out = translate_shim_error(
            "DEPENDENCY_MISSING",
            "pyserial missing",
            {"dep": "pyserial"},
            _make_shim(),
        )
        assert "pyserial" in out
        assert "[<extra>]" in out


class TestPassthroughForUnknownCodes:
    def test_unknown_code_returns_passthrough_string(self):
        out = translate_shim_error(
            "TOTALLY_NEW_ERROR_CODE", "something blew up", {}, _make_shim()
        )
        assert "TOTALLY_NEW_ERROR_CODE" in out
        assert "something blew up" in out

    def test_unknown_code_with_no_shim_does_not_crash(self):
        out = translate_shim_error("WHO_KNOWS", "boom", None, None)
        assert "WHO_KNOWS" in out
        assert "boom" in out


class TestTranslatorIsTotal:
    """The translator must NEVER throw — it sits on the unhappy path."""

    @pytest.mark.parametrize("code", list(_EXPECTED_SUBSTRINGS.keys()))
    def test_every_known_code_with_empty_details(self, code: str):
        out = translate_shim_error(code, "", {}, _make_shim())
        assert isinstance(out, str)
        assert len(out) > 0

    @pytest.mark.parametrize("code", list(_EXPECTED_SUBSTRINGS.keys()))
    def test_every_known_code_with_no_shim(self, code: str):
        out = translate_shim_error(code, "msg", {}, None)
        assert isinstance(out, str)
        assert len(out) > 0

    @pytest.mark.parametrize("code", list(_EXPECTED_SUBSTRINGS.keys()))
    def test_every_known_code_with_none_details(self, code: str):
        out = translate_shim_error(code, "msg", None, _make_shim())
        assert isinstance(out, str)
        assert len(out) > 0

    def test_handles_none_code_gracefully(self):
        out = translate_shim_error("", "boom", {}, None)
        assert isinstance(out, str)
        assert len(out) > 0


# ---------------------------------------------------------------------------
# Integration-style: wire ERROR envelope → _FilesProxy/_CommandsProxy → OSError
# ---------------------------------------------------------------------------


def _build_files_shim(rpc_return: dict) -> LocalPCShim:
    shim = LocalPCShim.__new__(LocalPCShim)
    shim.sandbox_id = "s1"
    shim.allowed_root = "/Users/test/ws"
    shim.platform = "darwin"
    shim.arch = "arm64"
    shim.machine_id = "m1"
    shim.capabilities = []
    shim._rpc = AsyncMock(return_value=rpc_return)
    shim.files = _FilesProxy(shim)
    shim.commands = _CommandsProxy(shim)
    return shim


class TestFilesProxyUsesTranslator:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_read_error_is_translated_not_raw_enum(self):
        envelope = {
            "type": "ERROR",
            "payload": {
                "code": "PATH_OUTSIDE_ALLOWED_ROOT",
                "message": "Path /etc/passwd is outside allowed root",
                "details": {"path": "/etc/passwd"},
            },
        }
        shim = _build_files_shim(envelope)
        with pytest.raises(OSError) as ei:
            await shim.files.read("/etc/passwd")
        msg = ei.value.args[0]
        # Must be the friendly translation, not the raw code.
        assert "PATH_OUTSIDE_ALLOWED_ROOT" not in msg
        assert "/Users/test/ws" in msg
        assert "/etc/passwd" in msg

    @pytest.mark.asyncio(loop_scope="session")
    async def test_write_error_is_translated(self):
        envelope = {
            "type": "ERROR",
            "payload": {
                "code": "PATH_RESERVED_NAME",
                "message": "CON is reserved",
                "details": {},
            },
        }
        shim = _build_files_shim(envelope)
        with pytest.raises(OSError) as ei:
            await shim.files.write("/Users/test/ws/CON", "hi")
        assert "Windows reserved name" in ei.value.args[0]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_stat_error_is_translated(self):
        envelope = {
            "type": "ERROR",
            "payload": {
                "code": "PATH_NOT_FOUND",
                "message": "no such path",
                "details": {"path": "/Users/test/ws/missing.txt"},
            },
        }
        shim = _build_files_shim(envelope)
        with pytest.raises(OSError) as ei:
            await shim.files.stat("/Users/test/ws/missing.txt")
        msg = ei.value.args[0]
        assert "doesn't exist" in msg
        assert "/Users/test/ws/missing.txt" in msg

    @pytest.mark.asyncio(loop_scope="session")
    async def test_list_error_is_translated(self):
        envelope = {
            "type": "ERROR",
            "payload": {
                "code": "PATH_OUTSIDE_ALLOWED_ROOT",
                "message": "outside root",
                "details": {"path": "/var"},
            },
        }
        shim = _build_files_shim(envelope)
        with pytest.raises(OSError) as ei:
            await shim.files.list("/var")
        assert "/Users/test/ws" in ei.value.args[0]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_delete_error_is_translated(self):
        envelope = {
            "type": "ERROR",
            "payload": {
                "code": "PATH_NOT_EMPTY",
                "message": "dir not empty",
                "details": {"path": "/Users/test/ws/cache"},
            },
        }
        shim = _build_files_shim(envelope)
        with pytest.raises(OSError) as ei:
            await shim.files.delete("/Users/test/ws/cache")
        assert "recursive=true" in ei.value.args[0]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_move_error_is_translated(self):
        envelope = {
            "type": "ERROR",
            "payload": {
                "code": "PATH_EXISTS",
                "message": "dst exists",
                "details": {"dst": "/Users/test/ws/out.json"},
            },
        }
        shim = _build_files_shim(envelope)
        with pytest.raises(OSError) as ei:
            await shim.files.move("/Users/test/ws/in.json", "/Users/test/ws/out.json")
        assert "overwrite=true" in ei.value.args[0]


class TestCommandsProxyUsesTranslator:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_run_error_is_translated_to_recovery_hint(self):
        envelope = {
            "type": "ERROR",
            "payload": {
                "code": "SHELL_NOT_AVAILABLE",
                "message": "bash not found",
                "details": {"requested_shell": "bash"},
            },
        }
        shim = _build_files_shim(envelope)
        with pytest.raises(RuntimeError) as ei:
            await shim.commands.run("ls", shell="bash")
        msg = ei.value.args[0]
        assert "SHELL_NOT_AVAILABLE" not in msg
        assert "bash" in msg
        assert "auto" in msg


class TestShimHelloMetadataFlowsThrough:
    """Sanity: the shim's HELLO metadata reaches the translator."""

    def test_allowed_root_from_real_hello_appears_in_message(self):
        hello = ShimHello(
            machine_id="m1",
            platform="windows",
            arch="x86_64",
            allowed_root="C:\\Users\\alice\\autogpt-workspace",
        )
        shim = LocalPCShim.__new__(LocalPCShim)
        shim.sandbox_id = "s1"
        shim.allowed_root = hello.allowed_root
        shim.platform = hello.platform
        shim.arch = hello.arch
        out = translate_shim_error(
            "PATH_OUTSIDE_ALLOWED_ROOT",
            "outside",
            {"path": "D:\\secrets"},
            shim,
        )
        assert "C:\\Users\\alice\\autogpt-workspace" in out
