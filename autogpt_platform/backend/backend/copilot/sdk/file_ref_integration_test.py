"""Integration tests for @@agptfile: reference expansion in tool calls.

These tests verify the end-to-end behaviour of the file reference protocol:
- Parsing @@agptfile: tokens from tool arguments
- Resolving local-filesystem paths (sdk_cwd / ephemeral)
- Expanding references inside the tool-call pipeline (_execute_tool_sync)
- The extended Read tool handler (workspace:// pass-through via session context)

No real LLM or database is required; workspace reads are stubbed where needed.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.sdk.file_ref import (
    FileRef,
    expand_file_refs_in_args,
    expand_file_refs_in_string,
    read_file_bytes,
    resolve_file_ref,
)
from backend.copilot.sdk.tool_adapter import _read_file_handler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(session_id: str = "integ-sess") -> MagicMock:
    s = MagicMock()
    s.session_id = session_id
    return s


# ---------------------------------------------------------------------------
# Local-file resolution (sdk_cwd)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_file_ref_local_path():
    """resolve_file_ref reads a real local file when it's within sdk_cwd."""
    with tempfile.TemporaryDirectory() as sdk_cwd:
        # Write a test file inside sdk_cwd
        test_file = os.path.join(sdk_cwd, "hello.txt")
        with open(test_file, "w") as f:
            f.write("line1\nline2\nline3\n")

        session = _make_session()
        with patch("backend.copilot.context._current_sdk_cwd") as mock_cwd_var:
            mock_cwd_var.get.return_value = sdk_cwd

            ref = FileRef(uri=test_file, start_line=None, end_line=None)
            content = await resolve_file_ref(ref, user_id="u1", session=session)

        assert content == "line1\nline2\nline3\n"


@pytest.mark.asyncio
async def test_resolve_file_ref_local_path_with_line_range():
    """resolve_file_ref respects line ranges for local files."""
    with tempfile.TemporaryDirectory() as sdk_cwd:
        test_file = os.path.join(sdk_cwd, "multi.txt")
        lines = [f"line{i}\n" for i in range(1, 11)]  # line1 … line10
        with open(test_file, "w") as f:
            f.writelines(lines)

        session = _make_session()
        with patch("backend.copilot.context._current_sdk_cwd") as mock_cwd_var:
            mock_cwd_var.get.return_value = sdk_cwd

            ref = FileRef(uri=test_file, start_line=3, end_line=5)
            content = await resolve_file_ref(ref, user_id="u1", session=session)

        assert content == "line3\nline4\nline5\n"


@pytest.mark.asyncio
async def test_resolve_file_ref_rejects_path_outside_sdk_cwd():
    """resolve_file_ref raises ValueError for paths outside sdk_cwd."""
    with tempfile.TemporaryDirectory() as sdk_cwd:
        with (
            patch("backend.copilot.context._current_sdk_cwd") as mock_cwd_var,
            patch("backend.copilot.context._current_sandbox") as mock_sandbox_var,
        ):
            mock_cwd_var.get.return_value = sdk_cwd
            mock_sandbox_var.get.return_value = None

            ref = FileRef(uri="/etc/passwd", start_line=None, end_line=None)
            with pytest.raises(ValueError, match="not allowed"):
                await resolve_file_ref(ref, user_id="u1", session=_make_session())


# ---------------------------------------------------------------------------
# expand_file_refs_in_string — integration with real files
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_string_with_real_file():
    """expand_file_refs_in_string replaces @@agptfile: token with actual content."""
    with tempfile.TemporaryDirectory() as sdk_cwd:
        test_file = os.path.join(sdk_cwd, "data.txt")
        with open(test_file, "w") as f:
            f.write("hello world\n")

        with patch("backend.copilot.context._current_sdk_cwd") as mock_cwd_var:
            mock_cwd_var.get.return_value = sdk_cwd

            result = await expand_file_refs_in_string(
                f"Content: @@agptfile:{test_file}",
                user_id="u1",
                session=_make_session(),
            )

        assert result == "Content: hello world\n"


@pytest.mark.asyncio
async def test_expand_string_missing_file_is_surfaced_inline():
    """Missing file ref yields [file-ref error: …] inline rather than raising."""
    with tempfile.TemporaryDirectory() as sdk_cwd:
        missing = os.path.join(sdk_cwd, "does_not_exist.txt")

        with patch("backend.copilot.context._current_sdk_cwd") as mock_cwd_var:
            mock_cwd_var.get.return_value = sdk_cwd

            result = await expand_file_refs_in_string(
                f"@@agptfile:{missing}",
                user_id="u1",
                session=_make_session(),
            )

        assert "[file-ref error:" in result
        assert "not found" in result.lower() or "not allowed" in result.lower()


# ---------------------------------------------------------------------------
# expand_file_refs_in_args — dict traversal with real files
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_args_replaces_file_ref_in_nested_dict():
    """Nested @@agptfile: references in args are fully expanded."""
    with tempfile.TemporaryDirectory() as sdk_cwd:
        file_a = os.path.join(sdk_cwd, "a.txt")
        file_b = os.path.join(sdk_cwd, "b.txt")
        with open(file_a, "w") as f:
            f.write("AAA")
        with open(file_b, "w") as f:
            f.write("BBB")

        with patch("backend.copilot.context._current_sdk_cwd") as mock_cwd_var:
            mock_cwd_var.get.return_value = sdk_cwd

            result = await expand_file_refs_in_args(
                {
                    "outer": {
                        "content_a": f"@@agptfile:{file_a}",
                        "content_b": f"start @@agptfile:{file_b} end",
                    },
                    "count": 42,
                },
                user_id="u1",
                session=_make_session(),
            )

        assert result["outer"]["content_a"] == "AAA"
        assert result["outer"]["content_b"] == "start BBB end"
        assert result["count"] == 42


# ---------------------------------------------------------------------------
# expand_file_refs_in_args — bare ref structured parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bare_ref_json_returns_parsed_dict():
    """Bare ref to a .json file returns parsed dict, not raw string."""
    with tempfile.TemporaryDirectory() as sdk_cwd:
        json_file = os.path.join(sdk_cwd, "data.json")
        with open(json_file, "w") as f:
            f.write('{"key": "value", "count": 42}')

        with patch("backend.copilot.context._current_sdk_cwd") as mock_cwd_var:
            mock_cwd_var.get.return_value = sdk_cwd

            result = await expand_file_refs_in_args(
                {"data": f"@@agptfile:{json_file}"},
                user_id="u1",
                session=_make_session(),
            )

        assert result["data"] == {"key": "value", "count": 42}


@pytest.mark.asyncio
async def test_bare_ref_csv_returns_parsed_table():
    """Bare ref to a .csv file returns list[list[str]] table."""
    with tempfile.TemporaryDirectory() as sdk_cwd:
        csv_file = os.path.join(sdk_cwd, "data.csv")
        with open(csv_file, "w") as f:
            f.write("Name,Score\nAlice,90\nBob,85")

        with patch("backend.copilot.context._current_sdk_cwd") as mock_cwd_var:
            mock_cwd_var.get.return_value = sdk_cwd

            result = await expand_file_refs_in_args(
                {"input": f"@@agptfile:{csv_file}"},
                user_id="u1",
                session=_make_session(),
            )

        assert result["input"] == [
            ["Name", "Score"],
            ["Alice", "90"],
            ["Bob", "85"],
        ]


@pytest.mark.asyncio
async def test_bare_ref_unknown_extension_returns_string():
    """Bare ref to a file with unknown extension returns plain string."""
    with tempfile.TemporaryDirectory() as sdk_cwd:
        txt_file = os.path.join(sdk_cwd, "readme.txt")
        with open(txt_file, "w") as f:
            f.write("plain text content")

        with patch("backend.copilot.context._current_sdk_cwd") as mock_cwd_var:
            mock_cwd_var.get.return_value = sdk_cwd

            result = await expand_file_refs_in_args(
                {"data": f"@@agptfile:{txt_file}"},
                user_id="u1",
                session=_make_session(),
            )

        assert result["data"] == "plain text content"
        assert isinstance(result["data"], str)


@pytest.mark.asyncio
async def test_bare_ref_invalid_json_falls_back_to_string():
    """Bare ref to a .json file with invalid JSON falls back to string."""
    with tempfile.TemporaryDirectory() as sdk_cwd:
        json_file = os.path.join(sdk_cwd, "bad.json")
        with open(json_file, "w") as f:
            f.write("not valid json {{{")

        with patch("backend.copilot.context._current_sdk_cwd") as mock_cwd_var:
            mock_cwd_var.get.return_value = sdk_cwd

            result = await expand_file_refs_in_args(
                {"data": f"@@agptfile:{json_file}"},
                user_id="u1",
                session=_make_session(),
            )

        assert result["data"] == "not valid json {{{"
        assert isinstance(result["data"], str)


@pytest.mark.asyncio
async def test_embedded_ref_always_returns_string_even_for_json():
    """Embedded ref (text around it) returns plain string, not parsed JSON."""
    with tempfile.TemporaryDirectory() as sdk_cwd:
        json_file = os.path.join(sdk_cwd, "data.json")
        with open(json_file, "w") as f:
            f.write('{"key": "value"}')

        with patch("backend.copilot.context._current_sdk_cwd") as mock_cwd_var:
            mock_cwd_var.get.return_value = sdk_cwd

            result = await expand_file_refs_in_args(
                {"data": f"prefix @@agptfile:{json_file} suffix"},
                user_id="u1",
                session=_make_session(),
            )

        assert isinstance(result["data"], str)
        assert result["data"].startswith("prefix ")
        assert result["data"].endswith(" suffix")


@pytest.mark.asyncio
async def test_bare_ref_yaml_returns_parsed_dict():
    """Bare ref to a .yaml file returns parsed dict."""
    with tempfile.TemporaryDirectory() as sdk_cwd:
        yaml_file = os.path.join(sdk_cwd, "config.yaml")
        with open(yaml_file, "w") as f:
            f.write("name: test\ncount: 42\n")

        with patch("backend.copilot.context._current_sdk_cwd") as mock_cwd_var:
            mock_cwd_var.get.return_value = sdk_cwd

            result = await expand_file_refs_in_args(
                {"config": f"@@agptfile:{yaml_file}"},
                user_id="u1",
                session=_make_session(),
            )

        assert result["config"] == {"name": "test", "count": 42}


@pytest.mark.asyncio
async def test_bare_ref_binary_with_line_range_ignores_range():
    """Bare ref to a binary file (.parquet) with line range parses the full file.

    Binary formats (parquet, xlsx) ignore line ranges — the full content is
    parsed and the range is silently dropped with a log warning.
    """
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")
    try:
        import pyarrow  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        pytest.skip("pyarrow not installed")

    with tempfile.TemporaryDirectory() as sdk_cwd:
        parquet_file = os.path.join(sdk_cwd, "data.parquet")
        import io as _io

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        buf = _io.BytesIO()
        df.to_parquet(buf, index=False)
        with open(parquet_file, "wb") as f:
            f.write(buf.getvalue())

        with patch("backend.copilot.context._current_sdk_cwd") as mock_cwd_var:
            mock_cwd_var.get.return_value = sdk_cwd

            # Line range [1-2] should be silently ignored for binary formats.
            result = await expand_file_refs_in_args(
                {"data": f"@@agptfile:{parquet_file}[1-2]"},
                user_id="u1",
                session=_make_session(),
            )

        # Full file is returned despite the line range.
        assert result["data"] == [["A", "B"], [1, 4], [2, 5], [3, 6]]


@pytest.mark.asyncio
async def test_bare_ref_toml_returns_parsed_dict():
    """Bare ref to a .toml file returns parsed dict."""
    with tempfile.TemporaryDirectory() as sdk_cwd:
        toml_file = os.path.join(sdk_cwd, "config.toml")
        with open(toml_file, "w") as f:
            f.write('name = "test"\ncount = 42\n')

        with patch("backend.copilot.context._current_sdk_cwd") as mock_cwd_var:
            mock_cwd_var.get.return_value = sdk_cwd

            result = await expand_file_refs_in_args(
                {"config": f"@@agptfile:{toml_file}"},
                user_id="u1",
                session=_make_session(),
            )

        assert result["config"] == {"name": "test", "count": 42}


# ---------------------------------------------------------------------------
# _read_file_handler — extended to accept workspace:// and local paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_file_handler_local_file():
    """_read_file_handler rejects files in sdk_cwd (use read_file MCP tool for those).

    read_tool_result is restricted to SDK-internal tool-results/tool-outputs paths
    via is_sdk_tool_path(). sdk_cwd files should be read via the read_file (e2b_file_tools)
    handler, not via read_tool_result.
    """
    with tempfile.TemporaryDirectory() as sdk_cwd:
        test_file = os.path.join(sdk_cwd, "read_test.txt")
        lines = [f"L{i}\n" for i in range(1, 6)]
        with open(test_file, "w") as f:
            f.writelines(lines)

        with (
            patch("backend.copilot.context._current_sdk_cwd") as mock_cwd_var,
            patch("backend.copilot.context._current_project_dir") as mock_proj_var,
            patch(
                "backend.copilot.sdk.tool_adapter.get_execution_context",
                return_value=("user-1", _make_session()),
            ),
        ):
            mock_cwd_var.get.return_value = sdk_cwd
            # No project_dir set — so is_sdk_tool_path returns False for sdk_cwd paths
            mock_proj_var.get.return_value = ""

            result = await _read_file_handler(
                {"file_path": test_file, "offset": 0, "limit": 5}
            )

        # sdk_cwd paths are NOT allowed via read_tool_result (use read_file instead)
        assert result["isError"]
        assert "not allowed" in result["content"][0]["text"].lower()


@pytest.mark.asyncio
async def test_read_file_handler_workspace_uri():
    """_read_file_handler handles workspace:// URIs via the workspace manager."""
    mock_session = _make_session()
    mock_manager = AsyncMock()
    mock_manager.read_file_by_id.return_value = b"workspace file content\nline two\n"

    with (
        patch(
            "backend.copilot.sdk.tool_adapter.get_execution_context",
            return_value=("user-1", mock_session),
        ),
        patch(
            "backend.copilot.sdk.file_ref.get_workspace_manager",
            new=AsyncMock(return_value=mock_manager),
        ),
    ):
        result = await _read_file_handler(
            {"file_path": "workspace://file-id-abc", "offset": 0, "limit": 10}
        )

    assert not result["isError"], result["content"][0]["text"]
    text = result["content"][0]["text"]
    assert "workspace file content" in text
    assert "line two" in text


@pytest.mark.asyncio
async def test_read_file_handler_workspace_uri_no_session():
    """_read_file_handler returns error when workspace:// is used without session."""
    with patch(
        "backend.copilot.sdk.tool_adapter.get_execution_context",
        return_value=(None, None),
    ):
        result = await _read_file_handler({"file_path": "workspace://some-id"})

    assert result["isError"]
    assert "session" in result["content"][0]["text"].lower()


@pytest.mark.asyncio
async def test_read_file_handler_access_denied():
    """_read_file_handler rejects paths outside allowed locations."""
    with (
        patch("backend.copilot.context._current_sdk_cwd") as mock_cwd,
        patch("backend.copilot.context._current_sandbox") as mock_sandbox,
        patch(
            "backend.copilot.sdk.tool_adapter.get_execution_context",
            return_value=("user-1", _make_session()),
        ),
    ):
        mock_cwd.get.return_value = "/tmp/safe-dir"
        mock_sandbox.get.return_value = None

        result = await _read_file_handler({"file_path": "/etc/passwd"})

    assert result["isError"]
    assert "not allowed" in result["content"][0]["text"].lower()


# ---------------------------------------------------------------------------
# read_file_bytes — workspace:///path (virtual path) and E2B sandbox branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_file_bytes_workspace_virtual_path():
    """workspace:///path resolves via manager.read_file (is_path=True path)."""
    session = _make_session()
    mock_manager = AsyncMock()
    mock_manager.read_file.return_value = b"virtual path content"

    with patch(
        "backend.copilot.sdk.file_ref.get_workspace_manager",
        new=AsyncMock(return_value=mock_manager),
    ):
        result = await read_file_bytes("workspace:///reports/q1.md", "user-1", session)

    assert result == b"virtual path content"
    mock_manager.read_file.assert_awaited_once_with("/reports/q1.md")


@pytest.mark.asyncio
async def test_read_file_bytes_e2b_sandbox_branch():
    """read_file_bytes reads from the E2B sandbox when a sandbox is active."""
    session = _make_session()
    mock_sandbox = AsyncMock()
    mock_sandbox.files.read.return_value = bytearray(b"sandbox content")

    with (
        patch("backend.copilot.context._current_sdk_cwd") as mock_cwd,
        patch("backend.copilot.context._current_sandbox") as mock_sandbox_var,
        patch("backend.copilot.context._current_project_dir") as mock_proj,
    ):
        mock_cwd.get.return_value = ""
        mock_sandbox_var.get.return_value = mock_sandbox
        mock_proj.get.return_value = ""

        result = await read_file_bytes("/home/user/script.sh", None, session)

    assert result == b"sandbox content"
    mock_sandbox.files.read.assert_awaited_once_with(
        "/home/user/script.sh", format="bytes"
    )


@pytest.mark.asyncio
async def test_read_file_bytes_e2b_path_escapes_sandbox_raises():
    """read_file_bytes raises ValueError for paths that escape the sandbox root."""
    session = _make_session()
    mock_sandbox = AsyncMock()

    with (
        patch("backend.copilot.context._current_sdk_cwd") as mock_cwd,
        patch("backend.copilot.context._current_sandbox") as mock_sandbox_var,
        patch("backend.copilot.context._current_project_dir") as mock_proj,
    ):
        mock_cwd.get.return_value = ""
        mock_sandbox_var.get.return_value = mock_sandbox
        mock_proj.get.return_value = ""

        with pytest.raises(ValueError, match="not allowed"):
            await read_file_bytes("/etc/passwd", None, session)
