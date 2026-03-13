"""Tests for the @@agptfile: reference protocol (file_ref.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.sdk.file_ref import (
    _MAX_BARE_REF_BYTES,
    _MAX_EXPAND_CHARS,
    FileRef,
    FileRefExpansionError,
    _apply_line_range,
    expand_file_refs_in_args,
    expand_file_refs_in_string,
    parse_file_ref,
)

# ---------------------------------------------------------------------------
# parse_file_ref
# ---------------------------------------------------------------------------


def test_parse_file_ref_workspace_id():
    ref = parse_file_ref("@@agptfile:workspace://abc123")
    assert ref == FileRef(uri="workspace://abc123", start_line=None, end_line=None)


def test_parse_file_ref_workspace_id_with_mime():
    ref = parse_file_ref("@@agptfile:workspace://abc123#text/plain")
    assert ref is not None
    assert ref.uri == "workspace://abc123#text/plain"
    assert ref.start_line is None


def test_parse_file_ref_workspace_path():
    ref = parse_file_ref("@@agptfile:workspace:///reports/q1.md")
    assert ref is not None
    assert ref.uri == "workspace:///reports/q1.md"


def test_parse_file_ref_with_line_range():
    ref = parse_file_ref("@@agptfile:workspace://abc123[10-50]")
    assert ref == FileRef(uri="workspace://abc123", start_line=10, end_line=50)


def test_parse_file_ref_local_path():
    ref = parse_file_ref("@@agptfile:/tmp/copilot-session/output.py[1-100]")
    assert ref is not None
    assert ref.uri == "/tmp/copilot-session/output.py"
    assert ref.start_line == 1
    assert ref.end_line == 100


def test_parse_file_ref_no_match():
    assert parse_file_ref("just a normal string") is None
    assert parse_file_ref("workspace://abc123") is None  # missing @@agptfile: prefix
    assert (
        parse_file_ref("@@agptfile:workspace://abc123 extra") is None
    )  # not full match


def test_parse_file_ref_strips_whitespace():
    ref = parse_file_ref("  @@agptfile:workspace://abc123  ")
    assert ref is not None
    assert ref.uri == "workspace://abc123"


def test_parse_file_ref_invalid_range_zero_start():
    assert parse_file_ref("@@agptfile:workspace://abc123[0-5]") is None


def test_parse_file_ref_invalid_range_end_less_than_start():
    assert parse_file_ref("@@agptfile:workspace://abc123[10-5]") is None


def test_parse_file_ref_invalid_range_zero_end():
    assert parse_file_ref("@@agptfile:workspace://abc123[1-0]") is None


# ---------------------------------------------------------------------------
# _apply_line_range
# ---------------------------------------------------------------------------


TEXT = "line1\nline2\nline3\nline4\nline5\n"


def test_apply_line_range_no_range():
    assert _apply_line_range(TEXT, None, None) == TEXT


def test_apply_line_range_start_only():
    result = _apply_line_range(TEXT, 3, None)
    assert result == "line3\nline4\nline5\n"


def test_apply_line_range_full():
    result = _apply_line_range(TEXT, 2, 4)
    assert result == "line2\nline3\nline4\n"


def test_apply_line_range_single_line():
    result = _apply_line_range(TEXT, 2, 2)
    assert result == "line2\n"


def test_apply_line_range_beyond_eof():
    result = _apply_line_range(TEXT, 4, 999)
    assert result == "line4\nline5\n"


# ---------------------------------------------------------------------------
# expand_file_refs_in_string
# ---------------------------------------------------------------------------


def _make_session(session_id: str = "sess-1") -> MagicMock:
    session = MagicMock()
    session.session_id = session_id
    return session


async def _resolve_always(ref: FileRef, _user_id: str | None, _session: object) -> str:
    """Stub resolver that returns the URI and range as a descriptive string."""
    if ref.start_line is not None:
        return f"content:{ref.uri}[{ref.start_line}-{ref.end_line}]"
    return f"content:{ref.uri}"


@pytest.mark.asyncio
async def test_expand_no_refs():
    result = await expand_file_refs_in_string(
        "no references here", user_id="u1", session=_make_session()
    )
    assert result == "no references here"


@pytest.mark.asyncio
async def test_expand_single_ref():
    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve_always),
    ):
        result = await expand_file_refs_in_string(
            "@@agptfile:workspace://abc123",
            user_id="u1",
            session=_make_session(),
        )
    assert result == "content:workspace://abc123"


@pytest.mark.asyncio
async def test_expand_ref_with_range():
    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve_always),
    ):
        result = await expand_file_refs_in_string(
            "@@agptfile:workspace://abc123[10-50]",
            user_id="u1",
            session=_make_session(),
        )
    assert result == "content:workspace://abc123[10-50]"


@pytest.mark.asyncio
async def test_expand_ref_embedded_in_text():
    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve_always),
    ):
        result = await expand_file_refs_in_string(
            "Here is the file: @@agptfile:workspace://abc123 — done",
            user_id="u1",
            session=_make_session(),
        )
    assert result == "Here is the file: content:workspace://abc123 — done"


@pytest.mark.asyncio
async def test_expand_multiple_refs():
    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve_always),
    ):
        result = await expand_file_refs_in_string(
            "@@agptfile:workspace://file1 and @@agptfile:workspace://file2[1-5]",
            user_id="u1",
            session=_make_session(),
        )
    assert result == "content:workspace://file1 and content:workspace://file2[1-5]"


@pytest.mark.asyncio
async def test_expand_invalid_range_zero_start_surfaces_inline():
    """expand_file_refs_in_string surfaces [file-ref error: ...] for zero-start ranges."""
    result = await expand_file_refs_in_string(
        "@@agptfile:workspace://abc123[0-5]",
        user_id="u1",
        session=_make_session(),
    )
    assert "[file-ref error:" in result
    assert "line numbers must be >= 1" in result


@pytest.mark.asyncio
async def test_expand_invalid_range_end_less_than_start_surfaces_inline():
    """expand_file_refs_in_string surfaces [file-ref error: ...] when end < start."""
    result = await expand_file_refs_in_string(
        "prefix @@agptfile:workspace://abc123[10-5] suffix",
        user_id="u1",
        session=_make_session(),
    )
    assert "[file-ref error:" in result
    assert "end line must be >= start line" in result
    assert "prefix" in result
    assert "suffix" in result


@pytest.mark.asyncio
async def test_expand_ref_error_surfaces_inline():
    async def _raise(*args, **kwargs):  # noqa: ARG001
        raise ValueError("file not found")

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_raise),
    ):
        result = await expand_file_refs_in_string(
            "@@agptfile:workspace://bad",
            user_id="u1",
            session=_make_session(),
        )
    assert "[file-ref error:" in result
    assert "file not found" in result


# ---------------------------------------------------------------------------
# expand_file_refs_in_args
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_args_flat():
    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve_always),
    ):
        result = await expand_file_refs_in_args(
            {"content": "@@agptfile:workspace://abc123", "other": 42},
            user_id="u1",
            session=_make_session(),
        )
    assert result["content"] == "content:workspace://abc123"
    assert result["other"] == 42


@pytest.mark.asyncio
async def test_expand_args_nested_dict():
    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve_always),
    ):
        result = await expand_file_refs_in_args(
            {"outer": {"inner": "@@agptfile:workspace://nested"}},
            user_id="u1",
            session=_make_session(),
        )
    assert result["outer"]["inner"] == "content:workspace://nested"


@pytest.mark.asyncio
async def test_expand_args_list():
    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve_always),
    ):
        result = await expand_file_refs_in_args(
            {
                "items": [
                    "@@agptfile:workspace://a",
                    "plain",
                    "@@agptfile:workspace://b[1-3]",
                ]
            },
            user_id="u1",
            session=_make_session(),
        )
    assert result["items"] == [
        "content:workspace://a",
        "plain",
        "content:workspace://b[1-3]",
    ]


@pytest.mark.asyncio
async def test_expand_args_empty():
    result = await expand_file_refs_in_args({}, user_id="u1", session=_make_session())
    assert result == {}


@pytest.mark.asyncio
async def test_expand_args_no_refs():
    result = await expand_file_refs_in_args(
        {"key": "no refs here", "num": 1},
        user_id="u1",
        session=_make_session(),
    )
    assert result == {"key": "no refs here", "num": 1}


@pytest.mark.asyncio
async def test_expand_args_raises_on_file_ref_error():
    """expand_file_refs_in_args raises FileRefExpansionError instead of passing
    the inline error string to the tool, blocking tool execution."""

    async def _raise(*args, **kwargs):  # noqa: ARG001
        raise ValueError("path does not exist")

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_raise),
    ):
        with pytest.raises(FileRefExpansionError) as exc_info:
            await expand_file_refs_in_args(
                {"prompt": "@@agptfile:/home/user/missing.txt"},
                user_id="u1",
                session=_make_session(),
            )
    assert "path does not exist" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Bare-ref structured parsing (infer_format + parse_file_content)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bare_ref_json_returns_parsed():
    """Bare ref to a .json file returns parsed dict, not a string."""

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return '{"key": "value"}'

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        result = await expand_file_refs_in_args(
            {"data": "@@agptfile:workspace:///report.json"},
            user_id="u1",
            session=_make_session(),
        )
    assert result["data"] == {"key": "value"}


@pytest.mark.asyncio
async def test_bare_ref_csv_returns_rows():
    """Bare ref to a .csv file returns list[list[str]]."""

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return "Name,Score\nAlice,90\nBob,85"

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        result = await expand_file_refs_in_args(
            {"data": "@@agptfile:workspace:///data.csv"},
            user_id="u1",
            session=_make_session(),
        )
    assert result["data"] == [["Name", "Score"], ["Alice", "90"], ["Bob", "85"]]


@pytest.mark.asyncio
async def test_bare_ref_unknown_extension_falls_back_to_string():
    """Bare ref with unknown extension returns plain string."""

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return "plain text content"

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        result = await expand_file_refs_in_args(
            {"data": "@@agptfile:workspace:///readme.txt"},
            user_id="u1",
            session=_make_session(),
        )
    assert result["data"] == "plain text content"


@pytest.mark.asyncio
async def test_bare_ref_invalid_json_falls_back_to_string():
    """Bare ref to .json file with invalid content falls back to string."""

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return "not valid json"

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        result = await expand_file_refs_in_args(
            {"data": "@@agptfile:workspace:///bad.json"},
            user_id="u1",
            session=_make_session(),
        )
    assert result["data"] == "not valid json"


@pytest.mark.asyncio
async def test_embedded_ref_stays_string_even_for_json():
    """Embedded ref (mixed with text) always returns plain string."""

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return '{"key": "value"}'

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        result = await expand_file_refs_in_args(
            {"data": "See: @@agptfile:workspace:///report.json"},
            user_id="u1",
            session=_make_session(),
        )
    # Embedded ref → inline expansion → plain string with content substituted
    assert isinstance(result["data"], str)
    assert '{"key": "value"}' in result["data"]


@pytest.mark.asyncio
async def test_bare_ref_oversized_raises_error():
    """Bare ref exceeding _MAX_BARE_REF_BYTES raises FileRefExpansionError."""

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return "x" * (_MAX_BARE_REF_BYTES + 1)

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        with pytest.raises(FileRefExpansionError, match="too large"):
            await expand_file_refs_in_args(
                {"data": "@@agptfile:workspace:///huge.json"},
                user_id="u1",
                session=_make_session(),
            )


# ---------------------------------------------------------------------------
# Schema-aware string bypass
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bare_ref_csv_returns_raw_string_when_schema_expects_string():
    """When the input schema declares the parameter as type: string,
    structured parsing is skipped and raw file content is returned."""
    csv_content = "Name,Score\nAlice,90\nBob,85"

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return csv_content

    schema = {
        "type": "object",
        "properties": {"data": {"type": "string"}},
        "required": ["data"],
    }

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        result = await expand_file_refs_in_args(
            {"data": "@@agptfile:workspace:///data.csv"},
            user_id="u1",
            session=_make_session(),
            input_schema=schema,
        )
    # Should return the raw CSV text, NOT a parsed list[list[str]]
    assert result["data"] == csv_content
    assert isinstance(result["data"], str)


@pytest.mark.asyncio
async def test_bare_ref_json_parses_when_schema_expects_object():
    """When the input schema declares the parameter as type: object,
    structured parsing is applied as usual."""

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return '{"key": "value"}'

    schema = {
        "type": "object",
        "properties": {"data": {"type": "object"}},
        "required": ["data"],
    }

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        result = await expand_file_refs_in_args(
            {"data": "@@agptfile:workspace:///report.json"},
            user_id="u1",
            session=_make_session(),
            input_schema=schema,
        )
    assert result["data"] == {"key": "value"}


@pytest.mark.asyncio
async def test_bare_ref_csv_parses_when_no_schema_provided():
    """Without input_schema, structured parsing proceeds as usual."""
    csv_content = "Name,Score\nAlice,90\nBob,85"

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return csv_content

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        result = await expand_file_refs_in_args(
            {"data": "@@agptfile:workspace:///data.csv"},
            user_id="u1",
            session=_make_session(),
        )
    assert result["data"] == [["Name", "Score"], ["Alice", "90"], ["Bob", "85"]]


# ---------------------------------------------------------------------------
# Per-file truncation and aggregate budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_per_file_truncation():
    """Content exceeding _MAX_EXPAND_CHARS is truncated with a marker."""
    oversized = "x" * (_MAX_EXPAND_CHARS + 100)

    async def _resolve_oversized(ref: FileRef, _uid: str | None, _s: object) -> str:
        return oversized

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve_oversized),
    ):
        result = await expand_file_refs_in_string(
            "@@agptfile:workspace://big-file",
            user_id="u1",
            session=_make_session(),
        )

    assert len(result) <= _MAX_EXPAND_CHARS + len("\n... [truncated]") + 10
    assert "[truncated]" in result


@pytest.mark.asyncio
async def test_expand_aggregate_budget_exhausted():
    """When the aggregate budget is exhausted, later refs get the budget message."""
    # Each file returns just under 300K; after ~4 files the 1M budget is used.
    big_chunk = "y" * 300_000

    async def _resolve_big(ref: FileRef, _uid: str | None, _s: object) -> str:
        return big_chunk

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve_big),
    ):
        # 5 refs @ 300K each = 1.5M → last ref(s) should hit the aggregate limit
        refs = " ".join(f"@@agptfile:workspace://f{i}" for i in range(5))
        result = await expand_file_refs_in_string(
            refs,
            user_id="u1",
            session=_make_session(),
        )

    assert "budget exhausted" in result
