"""Tests for the @@agptfile: reference protocol (file_ref.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pydantic
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
from backend.util.type import coerce_inputs_to_schema

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


@pytest.mark.asyncio
async def test_opaque_object_skips_inner_expansion():
    """When the schema declares a property as {type: "object"} with no
    properties, inner file refs are NOT expanded — they stay as-is for
    the tool to expand later with the correct nested schema."""

    schema = {
        "type": "object",
        "properties": {
            "block_id": {"type": "string"},
            "input_data": {
                "type": "object",
                "description": "Opaque block inputs",
            },
        },
    }

    # resolve_file_ref should NOT be called for the inner ref
    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=AssertionError("should not be called")),
    ):
        result = await expand_file_refs_in_args(
            {
                "block_id": "some-id",
                "input_data": {"text": "@@agptfile:workspace:///data.csv"},
            },
            user_id="u1",
            session=_make_session(),
            input_schema=schema,
        )
    # input_data should be returned unchanged
    assert result["input_data"]["text"] == "@@agptfile:workspace:///data.csv"


@pytest.mark.asyncio
async def test_nested_schema_propagation():
    """When the schema declares nested properties, the inner type
    information is used for expand/skip decisions."""
    csv_content = "Name,Score\nAlice,90\nBob,85"

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return csv_content

    schema = {
        "type": "object",
        "properties": {
            "input_data": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "rows": {"type": "array"},
                },
            },
        },
    }

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        result = await expand_file_refs_in_args(
            {
                "input_data": {
                    "text": "@@agptfile:workspace:///data.csv",
                    "rows": "@@agptfile:workspace:///data.csv",
                },
            },
            user_id="u1",
            session=_make_session(),
            input_schema=schema,
        )
    # string-typed field: raw CSV text
    assert result["input_data"]["text"] == csv_content
    # array-typed field: parsed rows
    assert result["input_data"]["rows"] == [
        ["Name", "Score"],
        ["Alice", "90"],
        ["Bob", "85"],
    ]


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


# ---------------------------------------------------------------------------
# Full format × schema-type matrix
# ---------------------------------------------------------------------------
# Each text format is tested against two schema types:
#   - type: string  → raw file content returned as-is
#   - no type / non-string → structured parsing applied
# This ensures CSV→string blocks get raw CSV, not json.dumps(list[list[str]]).

_FORMAT_SAMPLES: dict[str, tuple[str, str, object]] = {
    # format_label: (extension, raw_content, expected_parsed_value)
    "json": (
        ".json",
        '{"name": "Alice", "score": 90}',
        {"name": "Alice", "score": 90},
    ),
    "csv": (
        ".csv",
        "Name,Score\nAlice,90\nBob,85",
        [["Name", "Score"], ["Alice", "90"], ["Bob", "85"]],
    ),
    "tsv": (
        ".tsv",
        "Name\tScore\nAlice\t90\nBob\t85",
        [["Name", "Score"], ["Alice", "90"], ["Bob", "85"]],
    ),
    "jsonl": (
        ".jsonl",
        '{"a":1}\n{"a":2}',
        [["a"], [1], [2]],  # uniform dicts → table format
    ),
    "yaml": (
        ".yaml",
        "name: Alice\nscore: 90",
        {"name": "Alice", "score": 90},
    ),
    "toml": (
        ".toml",
        '[person]\nname = "Alice"\nscore = 90',
        {"person": {"name": "Alice", "score": 90}},
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", _FORMAT_SAMPLES.keys())
async def test_matrix_format_to_string_schema(fmt: str):
    """When the schema declares the parameter as type: string,
    every text format returns the raw file content as a plain string."""
    ext, raw_content, _ = _FORMAT_SAMPLES[fmt]

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return raw_content

    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
    }

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        result = await expand_file_refs_in_args(
            {"text": f"@@agptfile:workspace:///data{ext}"},
            user_id="u1",
            session=_make_session(),
            input_schema=schema,
        )
    assert isinstance(
        result["text"], str
    ), f"{fmt}: expected str, got {type(result['text'])}"
    assert result["text"] == raw_content, f"{fmt}: raw content mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", _FORMAT_SAMPLES.keys())
async def test_matrix_format_to_nonstring_schema(fmt: str):
    """Without a string schema constraint, every text format returns
    a structured (parsed) value."""
    ext, raw_content, expected_parsed = _FORMAT_SAMPLES[fmt]

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return raw_content

    # No input_schema → structured parsing is the default
    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        result = await expand_file_refs_in_args(
            {"data": f"@@agptfile:workspace:///data{ext}"},
            user_id="u1",
            session=_make_session(),
        )
    assert result["data"] == expected_parsed, f"{fmt}: parsed value mismatch"
    assert not isinstance(
        result["data"], str
    ), f"{fmt}: expected structured type, got str"


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", _FORMAT_SAMPLES.keys())
async def test_matrix_format_opaque_object_preserves_ref(fmt: str):
    """When the parameter is an opaque object (type: object, no properties),
    inner file refs are preserved for second-phase expansion."""
    ext, _, _ = _FORMAT_SAMPLES[fmt]

    schema = {
        "type": "object",
        "properties": {
            "input_data": {"type": "object"},
        },
    }

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=AssertionError("should not be called")),
    ):
        ref_token = f"@@agptfile:workspace:///data{ext}"
        result = await expand_file_refs_in_args(
            {"input_data": {"field": ref_token}},
            user_id="u1",
            session=_make_session(),
            input_schema=schema,
        )
    assert result["input_data"]["field"] == ref_token, f"{fmt}: ref should be preserved"


@pytest.mark.asyncio
async def test_matrix_mixed_fields_string_and_array():
    """A single call with both string-typed and array-typed fields:
    CSV→string returns raw text, CSV→array returns parsed rows."""
    csv_content = "A,B\n1,2\n3,4"

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return csv_content

    schema = {
        "type": "object",
        "properties": {
            "raw_text": {"type": "string"},
            "parsed_rows": {"type": "array"},
        },
    }

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        result = await expand_file_refs_in_args(
            {
                "raw_text": "@@agptfile:workspace:///data.csv",
                "parsed_rows": "@@agptfile:workspace:///data.csv",
            },
            user_id="u1",
            session=_make_session(),
            input_schema=schema,
        )
    assert isinstance(result["raw_text"], str)
    assert result["raw_text"] == csv_content
    assert result["parsed_rows"] == [["A", "B"], ["1", "2"], ["3", "4"]]


@pytest.mark.asyncio
async def test_matrix_second_phase_expansion_with_block_schema():
    """Simulates the two-phase expansion: first phase skips opaque input_data,
    second phase expands with the block's actual schema."""
    csv_content = "X,Y\n10,20"

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return csv_content

    # Phase 1: tool-level schema (input_data is opaque)
    tool_schema = {
        "type": "object",
        "properties": {
            "block_id": {"type": "string"},
            "input_data": {"type": "object"},
        },
    }

    ref_token = "@@agptfile:workspace:///data.csv"
    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=AssertionError("phase 1 should not expand")),
    ):
        phase1_result = await expand_file_refs_in_args(
            {"block_id": "some-id", "input_data": {"text": ref_token}},
            user_id="u1",
            session=_make_session(),
            input_schema=tool_schema,
        )
    # Ref should survive phase 1
    assert phase1_result["input_data"]["text"] == ref_token

    # Phase 2: block-level schema (text is string)
    block_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
        },
    }

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        phase2_result = await expand_file_refs_in_args(
            phase1_result["input_data"],
            user_id="u1",
            session=_make_session(),
            input_schema=block_schema,
        )
    # Phase 2 should return raw CSV since text is string-typed
    assert isinstance(phase2_result["text"], str)
    assert phase2_result["text"] == csv_content


# ---------------------------------------------------------------------------
# End-to-end: file ref expansion + coerce_inputs_to_schema
# ---------------------------------------------------------------------------
# This proves the full pipeline: expand_file_refs_in_args returns the right
# type, and coerce_inputs_to_schema does NOT re-serialize it (the bug that
# caused CSV→string blocks to receive json.dumps(list[list[str]])).


class _StringBlock(pydantic.BaseModel):
    """Simulates a block schema with a string-typed input (e.g. TextEncoderBlock)."""

    text: str


class _ListBlock(pydantic.BaseModel):
    """Simulates a block schema with a list-typed input (e.g. ConcatenateListsBlock)."""

    rows: list


@pytest.mark.asyncio
async def test_e2e_csv_to_string_block_no_json_dumps():
    """Full pipeline: CSV file ref → expand with string schema → coerce.
    The block receives the raw CSV text, NOT json.dumps(parsed_rows)."""
    csv_content = "Name,Score\nAlice,90\nBob,85"

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return csv_content

    block_schema = _StringBlock.model_json_schema()

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        expanded = await expand_file_refs_in_args(
            {"text": "@@agptfile:workspace:///data.csv"},
            user_id="u1",
            session=_make_session(),
            input_schema=block_schema,
        )

    # After expansion: should be raw CSV string
    assert isinstance(expanded["text"], str)
    assert expanded["text"] == csv_content

    # After coercion: should still be raw CSV string (not json.dumps of parsed data)
    coerce_inputs_to_schema(expanded, _StringBlock)
    assert expanded["text"] == csv_content
    assert "[[" not in expanded["text"], "CSV was parsed and json.dumps'd — the old bug"


@pytest.mark.asyncio
async def test_e2e_csv_to_list_block_parses():
    """Full pipeline: CSV file ref → expand without string schema → coerce.
    The block receives parsed rows."""
    csv_content = "A,B\n1,2"

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return csv_content

    block_schema = _ListBlock.model_json_schema()

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        expanded = await expand_file_refs_in_args(
            {"rows": "@@agptfile:workspace:///data.csv"},
            user_id="u1",
            session=_make_session(),
            input_schema=block_schema,
        )

    # After expansion: should be parsed rows
    assert expanded["rows"] == [["A", "B"], ["1", "2"]]

    # After coercion: list satisfies list annotation, no conversion needed
    coerce_inputs_to_schema(expanded, _ListBlock)
    assert expanded["rows"] == [["A", "B"], ["1", "2"]]


@pytest.mark.asyncio
async def test_e2e_json_to_string_block_returns_raw_json():
    """JSON file ref to a string-typed block input returns the raw JSON text,
    not the parsed dict."""
    json_content = '{"key": "value", "num": 42}'

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return json_content

    block_schema = _StringBlock.model_json_schema()

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        expanded = await expand_file_refs_in_args(
            {"text": "@@agptfile:workspace:///config.json"},
            user_id="u1",
            session=_make_session(),
            input_schema=block_schema,
        )

    assert isinstance(expanded["text"], str)
    assert expanded["text"] == json_content

    coerce_inputs_to_schema(expanded, _StringBlock)
    assert expanded["text"] == json_content


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt,ext", [("parquet", ".parquet"), ("xlsx", ".xlsx")])
async def test_e2e_binary_format_to_string_block_raises_error(fmt: str, ext: str):
    """Binary formats (parquet/xlsx) passed to a string-typed block input
    should raise FileRefExpansionError, not silently pass garbled bytes."""

    async def _resolve_bytes(uri, user_id, session):  # noqa: ARG001
        return b"\x00\x01\x02binary data"

    block_schema = _StringBlock.model_json_schema()

    with patch(
        "backend.copilot.sdk.file_ref.read_file_bytes",
        new=AsyncMock(side_effect=_resolve_bytes),
    ):
        with pytest.raises(FileRefExpansionError, match="Cannot use .* as text input"):
            await expand_file_refs_in_args(
                {"text": f"@@agptfile:workspace:///data{ext}"},
                user_id="u1",
                session=_make_session(),
                input_schema=block_schema,
            )


# ---------------------------------------------------------------------------
# E2E: JSONL × block type matrix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_jsonl_tabular_to_list_block():
    """JSONL with uniform dicts → list block: table format (header + rows),
    consistent with CSV/TSV/Parquet/Excel output."""
    jsonl_content = (
        '{"name":"apple","color":"red"}\n'
        '{"name":"banana","color":"yellow"}\n'
        '{"name":"cherry","color":"red"}'
    )

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return jsonl_content

    block_schema = _ListBlock.model_json_schema()

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        expanded = await expand_file_refs_in_args(
            {"rows": "@@agptfile:workspace:///data.jsonl"},
            user_id="u1",
            session=_make_session(),
            input_schema=block_schema,
        )

    # Table format: header row + data rows
    assert expanded["rows"] == [
        ["name", "color"],
        ["apple", "red"],
        ["banana", "yellow"],
        ["cherry", "red"],
    ]

    # After coercion: list[list] fits list annotation perfectly
    coerce_inputs_to_schema(expanded, _ListBlock)
    assert expanded["rows"][0] == ["name", "color"]


@pytest.mark.asyncio
async def test_e2e_jsonl_tabular_to_string_block():
    """JSONL → string block: raw JSONL text, no parsing."""
    jsonl_content = '{"name":"apple"}\n{"name":"banana"}'

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return jsonl_content

    block_schema = _StringBlock.model_json_schema()

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        expanded = await expand_file_refs_in_args(
            {"text": "@@agptfile:workspace:///data.jsonl"},
            user_id="u1",
            session=_make_session(),
            input_schema=block_schema,
        )

    assert isinstance(expanded["text"], str)
    assert expanded["text"] == jsonl_content

    coerce_inputs_to_schema(expanded, _StringBlock)
    assert expanded["text"] == jsonl_content


class _DictBlock(pydantic.BaseModel):
    """Simulates a block schema with a dict-typed input (e.g. FindInDictionaryBlock)."""

    data: dict


@pytest.mark.asyncio
async def test_e2e_jsonl_tabular_to_dict_block():
    """JSONL with uniform dicts → dict block: table format is coerced to
    an indexed dict {0: header, 1: row1, ...}, consistent with CSV."""
    jsonl_content = '{"name":"apple","color":"red"}\n{"name":"banana","color":"yellow"}'

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return jsonl_content

    block_schema = _DictBlock.model_json_schema()

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        expanded = await expand_file_refs_in_args(
            {"data": "@@agptfile:workspace:///data.jsonl"},
            user_id="u1",
            session=_make_session(),
            input_schema=block_schema,
        )

    # Table format from parser
    assert expanded["data"] == [
        ["name", "color"],
        ["apple", "red"],
        ["banana", "yellow"],
    ]

    # After coercion to dict: pydantic indexes the list
    coerce_inputs_to_schema(expanded, _DictBlock)
    assert isinstance(expanded["data"], dict)


@pytest.mark.asyncio
async def test_e2e_jsonl_heterogeneous_to_list_block():
    """JSONL with different keys → list block: returns list of dicts (no table)."""
    jsonl_content = '{"name":"apple"}\n{"color":"red"}\n{"size":3}'

    async def _resolve(ref, *a, **kw):  # noqa: ARG001
        return jsonl_content

    block_schema = _ListBlock.model_json_schema()

    with patch(
        "backend.copilot.sdk.file_ref.resolve_file_ref",
        new=AsyncMock(side_effect=_resolve),
    ):
        expanded = await expand_file_refs_in_args(
            {"rows": "@@agptfile:workspace:///data.jsonl"},
            user_id="u1",
            session=_make_session(),
            input_schema=block_schema,
        )

    # Heterogeneous dicts stay as list of dicts
    assert expanded["rows"] == [
        {"name": "apple"},
        {"color": "red"},
        {"size": 3},
    ]


# ---------------------------------------------------------------------------
# E2E: Parquet × block type matrix (requires pyarrow)
# ---------------------------------------------------------------------------

_PARQUET_AVAILABLE = True
try:
    import pyarrow  # noqa: F401
except ImportError:
    _PARQUET_AVAILABLE = False


def _make_parquet_bytes() -> bytes:
    """Create a small parquet file in memory."""
    import io

    import pandas as pd

    df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [90, 85]})
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


@pytest.mark.skipif(not _PARQUET_AVAILABLE, reason="pyarrow not installed")
@pytest.mark.asyncio
async def test_e2e_parquet_to_list_block():
    """Parquet → list block: table format (header + rows)."""
    parquet_bytes = _make_parquet_bytes()

    async def _resolve_bytes(uri, user_id, session):  # noqa: ARG001
        return parquet_bytes

    block_schema = _ListBlock.model_json_schema()

    with patch(
        "backend.copilot.sdk.file_ref.read_file_bytes",
        new=AsyncMock(side_effect=_resolve_bytes),
    ):
        expanded = await expand_file_refs_in_args(
            {"rows": "@@agptfile:workspace:///data.parquet"},
            user_id="u1",
            session=_make_session(),
            input_schema=block_schema,
        )

    assert expanded["rows"] == [["Name", "Score"], ["Alice", 90], ["Bob", 85]]


@pytest.mark.skipif(not _PARQUET_AVAILABLE, reason="pyarrow not installed")
@pytest.mark.asyncio
async def test_e2e_parquet_to_dict_block():
    """Parquet → dict block: table format coerced to indexed dict."""
    parquet_bytes = _make_parquet_bytes()

    async def _resolve_bytes(uri, user_id, session):  # noqa: ARG001
        return parquet_bytes

    block_schema = _DictBlock.model_json_schema()

    with patch(
        "backend.copilot.sdk.file_ref.read_file_bytes",
        new=AsyncMock(side_effect=_resolve_bytes),
    ):
        expanded = await expand_file_refs_in_args(
            {"data": "@@agptfile:workspace:///data.parquet"},
            user_id="u1",
            session=_make_session(),
            input_schema=block_schema,
        )

    # Parser returns table format
    assert expanded["data"] == [["Name", "Score"], ["Alice", 90], ["Bob", 85]]

    # After coercion to dict: pydantic indexes the list
    coerce_inputs_to_schema(expanded, _DictBlock)
    assert isinstance(expanded["data"], dict)
