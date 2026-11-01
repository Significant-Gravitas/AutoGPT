"""Tests for file_content_parser — format inference and structured parsing."""

import io
import json
import struct
import zipfile
from unittest.mock import patch

import pytest

from backend.util import file_content_parser
from backend.util.file_content_parser import (
    BINARY_FORMATS,
    infer_format_from_uri,
    parse_file_content,
)
from backend.util.zip_member import ZipMemberError

# ---------------------------------------------------------------------------
# infer_format_from_uri
# ---------------------------------------------------------------------------


class TestInferFormat:
    @pytest.mark.parametrize(
        "uri, expected",
        [
            # extension
            ("/home/user/data.json", "json"),
            ("/tmp/events.jsonl", "jsonl"),
            ("/tmp/events.ndjson", "jsonl"),
            ("workspace:///reports/sales.csv", "csv"),
            ("/home/user/data.tsv", "tsv"),
            ("/home/user/config.yaml", "yaml"),
            ("/home/user/config.yml", "yaml"),
            ("/home/user/config.toml", "toml"),
            ("/data/table.parquet", "parquet"),
            ("/data/spreadsheet.xlsx", "xlsx"),
            # Legacy .xls is mapped so callers can produce a helpful error.
            ("/data/old_spreadsheet.xls", "xls"),
            ("/data/FILE.JSON", "json"),
            ("/data/FILE.CSV", "csv"),
            ("/home/user/\u30c7\u30fc\u30bf.json", "json"),
            ("/home/user/\u00e9t\u00e9.csv", "csv"),
            ("/home/user/readme.txt", None),
            ("workspace://abc123", None),
            # MIME
            ("workspace://abc123#application/json", "json"),
            ("workspace://abc123#text/csv", "csv"),
            ("workspace://abc123#text/tab-separated-values", "tsv"),
            ("workspace://abc123#application/x-ndjson", "jsonl"),
            ("workspace://abc123#application/x-yaml", "yaml"),
            (
                "workspace://abc123#application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "xlsx",
            ),
            ("workspace://abc123#application/vnd.apache.parquet", "parquet"),
            ("workspace://abc123#text/plain", None),
            # An unknown MIME falls through to the extension.
            ("workspace:///data.csv#text/plain", "csv"),
            # A known MIME wins over the extension.
            ("workspace:///file.txt#application/json", "json"),
        ],
    )
    def test_infer_format_from_uri(self, uri, expected):
        assert infer_format_from_uri(uri) == expected


# ---------------------------------------------------------------------------
# parse_file_content — JSON
# ---------------------------------------------------------------------------


class TestParseJson:
    def test_array(self):
        result = parse_file_content("[1, 2, 3]", "json")
        assert result == [1, 2, 3]

    def test_object(self):
        result = parse_file_content('{"key": "value"}', "json")
        assert result == {"key": "value"}

    def test_nested(self):
        content = json.dumps({"rows": [[1, 2], [3, 4]]})
        result = parse_file_content(content, "json")
        assert result == {"rows": [[1, 2], [3, 4]]}

    def test_scalar_string_stays_as_string(self):
        result = parse_file_content('"hello"', "json")
        assert result == '"hello"'  # original content, not parsed

    def test_scalar_number_stays_as_string(self):
        result = parse_file_content("42", "json")
        assert result == "42"

    def test_scalar_boolean_stays_as_string(self):
        result = parse_file_content("true", "json")
        assert result == "true"

    def test_null_stays_as_string(self):
        result = parse_file_content("null", "json")
        assert result == "null"

    def test_invalid_json_fallback(self):
        content = "not json at all"
        result = parse_file_content(content, "json")
        assert result == content

    def test_empty_string_fallback(self):
        result = parse_file_content("", "json")
        assert result == ""

    def test_bytes_input_decoded(self):
        result = parse_file_content(b"[1, 2, 3]", "json")
        assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# parse_file_content — JSONL
# ---------------------------------------------------------------------------


class TestParseJsonl:
    def test_tabular_uniform_dicts_to_table_format(self):
        """JSONL with uniform dict keys → table format (header + rows),
        consistent with CSV/TSV/Parquet/Excel output."""
        content = '{"name":"apple","color":"red"}\n{"name":"banana","color":"yellow"}\n{"name":"cherry","color":"red"}'
        result = parse_file_content(content, "jsonl")
        assert result == [
            ["name", "color"],
            ["apple", "red"],
            ["banana", "yellow"],
            ["cherry", "red"],
        ]

    def test_tabular_single_key_dicts(self):
        """JSONL with single-key uniform dicts → table format."""
        content = '{"a": 1}\n{"a": 2}\n{"a": 3}'
        result = parse_file_content(content, "jsonl")
        assert result == [["a"], [1], [2], [3]]

    def test_tabular_blank_lines_skipped(self):
        content = '{"a": 1}\n\n{"a": 2}\n'
        result = parse_file_content(content, "jsonl")
        assert result == [["a"], [1], [2]]

    def test_heterogeneous_dicts_stay_as_list(self):
        """JSONL with different keys across objects → list of dicts (no table)."""
        content = '{"name":"apple"}\n{"color":"red"}\n{"size":3}'
        result = parse_file_content(content, "jsonl")
        assert result == [{"name": "apple"}, {"color": "red"}, {"size": 3}]

    def test_partially_overlapping_keys_stay_as_list(self):
        """JSONL dicts with partially overlapping keys → list of dicts."""
        content = '{"name":"apple","color":"red"}\n{"name":"banana","size":"medium"}'
        result = parse_file_content(content, "jsonl")
        assert result == [
            {"name": "apple", "color": "red"},
            {"name": "banana", "size": "medium"},
        ]

    def test_mixed_types_stay_as_list(self):
        """JSONL with non-dict lines → list of parsed values (no table)."""
        content = '1\n"hello"\n[1,2]\n'
        result = parse_file_content(content, "jsonl")
        assert result == [1, "hello", [1, 2]]

    def test_mixed_dicts_and_non_dicts_stay_as_list(self):
        """JSONL mixing dicts and non-dicts → list of parsed values."""
        content = '{"a": 1}\n42\n{"b": 2}'
        result = parse_file_content(content, "jsonl")
        assert result == [{"a": 1}, 42, {"b": 2}]

    def test_tabular_preserves_key_order(self):
        """Table header should follow the key order of the first object."""
        content = '{"z": 1, "a": 2}\n{"z": 3, "a": 4}'
        result = parse_file_content(content, "jsonl")
        assert result[0] == ["z", "a"]  # order from first object
        assert result[1] == [1, 2]
        assert result[2] == [3, 4]

    def test_single_dict_stays_as_list(self):
        """Single-line JSONL with one dict → [dict], NOT a table.
        Tabular detection requires ≥2 dicts to avoid vacuously true all()."""
        content = '{"a": 1, "b": 2}'
        result = parse_file_content(content, "jsonl")
        assert result == [{"a": 1, "b": 2}]

    def test_tabular_with_none_values(self):
        """Uniform keys but some null values → table with None cells."""
        content = '{"name":"apple","color":"red"}\n{"name":"banana","color":null}'
        result = parse_file_content(content, "jsonl")
        assert result == [
            ["name", "color"],
            ["apple", "red"],
            ["banana", None],
        ]

    def test_empty_file_fallback(self):
        result = parse_file_content("", "jsonl")
        assert result == ""

    def test_all_blank_lines_fallback(self):
        result = parse_file_content("\n\n\n", "jsonl")
        assert result == "\n\n\n"

    def test_invalid_line_fallback(self):
        content = '{"a": 1}\nnot json\n'
        result = parse_file_content(content, "jsonl")
        assert result == content  # fallback


# ---------------------------------------------------------------------------
# parse_file_content — CSV
# ---------------------------------------------------------------------------


class TestParseCsv:
    def test_basic(self):
        content = "Name,Score\nAlice,90\nBob,85"
        result = parse_file_content(content, "csv")
        assert result == [["Name", "Score"], ["Alice", "90"], ["Bob", "85"]]

    def test_quoted_fields(self):
        content = 'Name,Bio\nAlice,"Loves, commas"\nBob,Simple'
        result = parse_file_content(content, "csv")
        assert result[1] == ["Alice", "Loves, commas"]

    def test_single_column_fallback(self):
        # Only 1 column — not tabular enough.
        content = "Name\nAlice\nBob"
        result = parse_file_content(content, "csv")
        assert result == content

    def test_empty_rows_skipped(self):
        content = "A,B\n\n1,2\n\n3,4"
        result = parse_file_content(content, "csv")
        assert result == [["A", "B"], ["1", "2"], ["3", "4"]]

    def test_empty_file_fallback(self):
        result = parse_file_content("", "csv")
        assert result == ""

    def test_utf8_bom(self):
        """CSV with a UTF-8 BOM should parse correctly (BOM stripped by decode)."""
        bom = "\ufeff"
        content = bom + "Name,Score\nAlice,90\nBob,85"
        result = parse_file_content(content, "csv")
        # The BOM may be part of the first header cell; ensure rows are still parsed.
        assert len(result) == 3
        assert result[1] == ["Alice", "90"]
        assert result[2] == ["Bob", "85"]


# ---------------------------------------------------------------------------
# parse_file_content — TSV
# ---------------------------------------------------------------------------


class TestParseTsv:
    def test_basic(self):
        content = "Name\tScore\nAlice\t90\nBob\t85"
        result = parse_file_content(content, "tsv")
        assert result == [["Name", "Score"], ["Alice", "90"], ["Bob", "85"]]

    def test_single_column_fallback(self):
        content = "Name\nAlice\nBob"
        result = parse_file_content(content, "tsv")
        assert result == content


# ---------------------------------------------------------------------------
# parse_file_content — YAML
# ---------------------------------------------------------------------------

KIB = 1024
MIB = 1024 * KIB


def _yaml_expanding_to(expanded: int, *, padding: int) -> str:
    """YAML whose text is about ``padding`` bytes plus a few KiB, and whose
    ``uses`` list expands to about ``expanded`` bytes through an alias."""
    uses = ", ".join(["*b"] * (expanded // 1000))
    return f"pad: {'y' * padding}\nbase: &b [{'x' * 1000}]\nuses: [{uses}]"


class TestParseYaml:
    def test_list(self):
        content = "- apple\n- banana\n- cherry"
        result = parse_file_content(content, "yaml")
        assert result == ["apple", "banana", "cherry"]

    def test_dict(self):
        content = "name: Alice\nage: 30"
        result = parse_file_content(content, "yaml")
        assert result == {"name": "Alice", "age": 30}

    def test_nested(self):
        content = "users:\n  - name: Alice\n  - name: Bob"
        result = parse_file_content(content, "yaml")
        assert result == {"users": [{"name": "Alice"}, {"name": "Bob"}]}

    def test_scalar_stays_as_string(self):
        result = parse_file_content("hello world", "yaml")
        assert result == "hello world"

    def test_invalid_yaml_fallback(self):
        content = ":\n  :\n    invalid: - -"
        result = parse_file_content(content, "yaml")
        # Malformed YAML should fall back to the original string, not raise.
        assert result == content

    def test_anchors_and_aliases_parse(self):
        content = "base: &b {x: 1}\nuse: *b"
        result = parse_file_content(content, "yaml")
        assert result == {"base": {"x": 1}, "use": {"x": 1}}

    def test_yaml_expanding_past_the_limit_falls_back_to_text(self, monkeypatch):
        monkeypatch.setattr(
            "backend.util.file_content_parser.MAX_YAML_EXPANDED_BYTES", 1024
        )
        uses = ", ".join(["*b"] * 40)
        content = f"base: &b [xxxxxxxx, xxxxxxxx, xxxxxxxx]\nuses: [{uses}]"

        assert parse_file_content(content, "yaml") == content
        with pytest.raises(ValueError, match="expands to more than 1 KiB"):
            parse_file_content(content, "yaml", strict=True)

    def test_yaml_that_refers_to_itself_falls_back_to_text(self):
        content = "items: &a [1, *a]"

        assert parse_file_content(content, "yaml") == content

    @pytest.mark.parametrize(
        "padding, expanded",
        [
            pytest.param(0, 900 * KIB, id="small file under the 1 MiB minimum"),
            pytest.param(256 * KIB, 1536 * KIB, id="256 KiB file under 8x its size"),
        ],
    )
    def test_yaml_within_its_expanded_budget_parses(self, padding, expanded):
        content = _yaml_expanding_to(expanded, padding=padding)

        assert isinstance(parse_file_content(content, "yaml"), dict)

    @pytest.mark.parametrize(
        "padding, expanded",
        [
            pytest.param(0, 2 * MIB, id="small file over the 1 MiB minimum"),
            pytest.param(256 * KIB, 3 * MIB, id="256 KiB file over 8x its size"),
        ],
    )
    def test_yaml_past_its_expanded_budget_falls_back_to_text(self, padding, expanded):
        content = _yaml_expanding_to(expanded, padding=padding)

        assert parse_file_content(content, "yaml") == content
        with pytest.raises(ValueError, match="expands to more than"):
            parse_file_content(content, "yaml", strict=True)


# ---------------------------------------------------------------------------
# parse_file_content — TOML
# ---------------------------------------------------------------------------


class TestParseToml:
    def test_basic(self):
        content = '[server]\nhost = "localhost"\nport = 8080'
        result = parse_file_content(content, "toml")
        assert result == {"server": {"host": "localhost", "port": 8080}}

    def test_flat(self):
        content = 'name = "test"\ncount = 42'
        result = parse_file_content(content, "toml")
        assert result == {"name": "test", "count": 42}

    def test_empty_string_returns_empty_dict(self):
        result = parse_file_content("", "toml")
        assert result == {}

    def test_invalid_toml_fallback(self):
        result = parse_file_content("not = [valid toml", "toml")
        assert result == "not = [valid toml"


# ---------------------------------------------------------------------------
# parse_file_content — Parquet (binary)
# ---------------------------------------------------------------------------


try:
    import pyarrow as _pa  # noqa: F401  # pyright: ignore[reportMissingImports]

    _has_pyarrow = True
except ImportError:
    _has_pyarrow = False


@pytest.mark.skipif(not _has_pyarrow, reason="pyarrow not installed")
class TestParseParquet:
    @pytest.fixture
    def parquet_bytes(self) -> bytes:
        import pandas as pd

        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [90, 85]})
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return buf.getvalue()

    def test_basic(self, parquet_bytes: bytes):
        result = parse_file_content(parquet_bytes, "parquet")
        assert result == [["Name", "Score"], ["Alice", 90], ["Bob", 85]]

    def test_string_input_fallback(self):
        # Parquet is binary — string input can't be parsed.
        result = parse_file_content("not parquet", "parquet")
        assert result == "not parquet"

    def test_invalid_bytes_fallback(self):
        result = parse_file_content(b"not parquet bytes", "parquet")
        assert result == b"not parquet bytes"

    def test_empty_bytes_fallback(self):
        """Empty binary input should return the empty bytes, not crash."""
        result = parse_file_content(b"", "parquet")
        assert result == b""

    def test_nan_replaced_with_none(self):
        """NaN values in Parquet must become None for JSON serializability."""
        import math

        import pandas as pd

        df = pd.DataFrame({"A": [1.0, float("nan"), 3.0], "B": ["x", None, "z"]})
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        result = parse_file_content(buf.getvalue(), "parquet")
        # Row with NaN in float col → None
        assert result[2][0] is None  # float NaN → None
        assert result[2][1] is None  # str None → None
        # Ensure no NaN leaks
        for row in result[1:]:
            for cell in row:
                if isinstance(cell, float):
                    assert not math.isnan(cell), f"NaN leaked: {row}"


# ---------------------------------------------------------------------------
# parse_file_content — Excel (binary)
# ---------------------------------------------------------------------------


class TestParseExcel:
    @pytest.fixture
    def xlsx_bytes(self) -> bytes:
        import pandas as pd

        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [90, 85]})
        buf = io.BytesIO()
        df.to_excel(buf, index=False)  # type: ignore[arg-type]  # BytesIO is a valid target
        return buf.getvalue()

    def test_basic(self, xlsx_bytes: bytes):
        result = parse_file_content(xlsx_bytes, "xlsx")
        assert result == [["Name", "Score"], ["Alice", 90], ["Bob", 85]]

    def test_string_input_fallback(self):
        result = parse_file_content("not xlsx", "xlsx")
        assert result == "not xlsx"

    def test_invalid_bytes_fallback(self):
        result = parse_file_content(b"not xlsx bytes", "xlsx")
        assert result == b"not xlsx bytes"

    def test_empty_bytes_fallback(self):
        """Empty binary input should return the empty bytes, not crash."""
        result = parse_file_content(b"", "xlsx")
        assert result == b""

    def test_nan_replaced_with_none(self):
        """NaN values in float columns must become None for JSON serializability."""
        import math

        import pandas as pd

        df = pd.DataFrame({"A": [1.0, float("nan"), 3.0], "B": ["x", "y", None]})
        buf = io.BytesIO()
        df.to_excel(buf, index=False)  # type: ignore[arg-type]
        result = parse_file_content(buf.getvalue(), "xlsx")
        # Row with NaN in float col → None, not float('nan')
        assert result[2][0] is None  # float NaN → None
        assert result[3][1] is None  # str None → None
        # Ensure no NaN leaks
        for row in result[1:]:  # skip header
            for cell in row:
                if isinstance(cell, float):
                    assert not math.isnan(cell), f"NaN leaked: {row}"


# ---------------------------------------------------------------------------
# parse_file_content — unknown format / fallback
# ---------------------------------------------------------------------------


class TestFallback:
    def test_unknown_format_returns_content(self):
        result = parse_file_content("hello world", "xml")
        assert result == "hello world"

    def test_none_format_returns_content(self):
        # Shouldn't normally be called with unrecognised format, but must not crash.
        result = parse_file_content("hello", "unknown_format")
        assert result == "hello"


# ---------------------------------------------------------------------------
# BINARY_FORMATS
# ---------------------------------------------------------------------------


class TestBinaryFormats:
    def test_parquet_is_binary(self):
        assert "parquet" in BINARY_FORMATS

    def test_xlsx_is_binary(self):
        assert "xlsx" in BINARY_FORMATS

    def test_text_formats_not_binary(self):
        for fmt in ("json", "jsonl", "csv", "tsv", "yaml", "toml"):
            assert fmt not in BINARY_FORMATS


# ---------------------------------------------------------------------------
# MIME mapping
# ---------------------------------------------------------------------------


class TestMimeMapping:
    def test_application_yaml(self):
        assert infer_format_from_uri("workspace://abc123#application/yaml") == "yaml"


# ---------------------------------------------------------------------------
# parse_file_content — Excel unpacked size
# ---------------------------------------------------------------------------

# The .xlsx caps: 50 MiB per member, 200 MiB per workbook.
_XLSX_MEMBER_CAP = 50 * 1024 * 1024


class TestXlsxUnpackedSize:
    @pytest.fixture
    def xlsx_bytes(self) -> bytes:
        from openpyxl import Workbook

        workbook = Workbook()
        sheet = workbook.active
        assert sheet is not None
        for row in (["Name", "Score"], ["Alice", 90], ["Bob", 85]):
            sheet.append(row)
        buf = io.BytesIO()
        workbook.save(buf)
        return buf.getvalue()

    @staticmethod
    def _with_member(xlsx: bytes, size: int) -> bytes:
        """Append a member of ``size`` zero bytes, written in pieces."""
        buf = io.BytesIO(xlsx)
        with (
            zipfile.ZipFile(buf, "a", zipfile.ZIP_DEFLATED) as archive,
            archive.open("xl/media/image1.png", "w") as member,
        ):
            for start in range(0, size, 4 * 1024 * 1024):
                member.write(b"\0" * min(4 * 1024 * 1024, size - start))
        return buf.getvalue()

    def test_a_member_past_the_cap_is_refused_unread(self, xlsx_bytes: bytes):
        data = self._with_member(xlsx_bytes, _XLSX_MEMBER_CAP + 1)
        assert len(data) < 1_000_000

        with (
            patch(
                "pandas.read_excel",
                side_effect=AssertionError("the workbook was read"),
            ) as read_excel,
            pytest.raises(ValueError, match="unpacks to more than"),
        ):
            parse_file_content(data, "xlsx", strict=True)
        read_excel.assert_not_called()

    def test_a_member_past_the_cap_falls_back_to_the_raw_bytes(self, xlsx_bytes: bytes):
        data = self._with_member(xlsx_bytes, _XLSX_MEMBER_CAP + 1)

        assert parse_file_content(data, "xlsx") == data

    def test_a_member_at_the_cap_is_read(self, xlsx_bytes: bytes):
        data = self._with_member(xlsx_bytes, _XLSX_MEMBER_CAP)

        result = parse_file_content(data, "xlsx", strict=True)
        assert result == [["Name", "Score"], ["Alice", 90], ["Bob", 85]]

    def test_members_together_past_the_total_are_refused(
        self, xlsx_bytes: bytes, monkeypatch
    ):
        monkeypatch.setattr(file_content_parser, "MAX_XLSX_TOTAL_BYTES", 1_000)

        with pytest.raises(ValueError, match="archive unpacks to more than 1000"):
            parse_file_content(xlsx_bytes, "xlsx", strict=True)

    def test_a_member_past_the_cap_is_reported_once(self, xlsx_bytes: bytes):
        data = self._with_member(xlsx_bytes, _XLSX_MEMBER_CAP + 1)

        with patch.object(file_content_parser.sentry_sdk, "capture_message") as report:
            parse_file_content(data, "xlsx")

        report.assert_called_once()
        (message,), kwargs = report.call_args
        assert message == "xlsx workbook over unpack limit"
        assert kwargs["level"] == "warning"
        assert kwargs["tags"] == {
            "xlsx_limit": "MAX_XLSX_MEMBER_BYTES",
            "xlsx_source": "file_content_parser",
        }
        assert kwargs["extras"]["limit_bytes"] == _XLSX_MEMBER_CAP
        assert kwargs["extras"]["unpacked_bytes"] > _XLSX_MEMBER_CAP

    def test_members_together_past_the_total_are_reported_once(
        self, xlsx_bytes: bytes, monkeypatch
    ):
        monkeypatch.setattr(file_content_parser, "MAX_XLSX_TOTAL_BYTES", 1_000)

        with patch.object(file_content_parser.sentry_sdk, "capture_message") as report:
            parse_file_content(xlsx_bytes, "xlsx")

        report.assert_called_once()
        assert report.call_args.kwargs["tags"]["xlsx_limit"] == "MAX_XLSX_TOTAL_BYTES"
        assert report.call_args.kwargs["extras"]["limit_bytes"] == 1_000
        assert report.call_args.kwargs["extras"]["unpacked_bytes"] > 1_000

    @staticmethod
    def _with_encrypted_flag(xlsx: bytes) -> bytes:
        """Set the encrypted flag on every member, in both of its headers."""
        out = bytearray(xlsx)
        with zipfile.ZipFile(io.BytesIO(xlsx)) as archive:
            entry = archive.start_dir
            for info in archive.infolist():
                out[info.header_offset + 6] |= 0x1
                out[entry + 8] |= 0x1
                name_len, extra_len, comment_len = struct.unpack_from(
                    "<HHH", out, entry + 28
                )
                entry += 46 + name_len + extra_len + comment_len
        return bytes(out)

    def test_an_encrypted_member_is_refused(self, xlsx_bytes: bytes):
        data = self._with_encrypted_flag(xlsx_bytes)

        with pytest.raises(ZipMemberError, match="is encrypted"):
            parse_file_content(data, "xlsx", strict=True)

    def test_an_encrypted_member_falls_back_to_the_raw_bytes(self, xlsx_bytes: bytes):
        data = self._with_encrypted_flag(xlsx_bytes)

        assert parse_file_content(data, "xlsx") == data

    def test_a_workbook_within_the_caps_is_not_reported(self, xlsx_bytes: bytes):
        with patch.object(file_content_parser.sentry_sdk, "capture_message") as report:
            parse_file_content(xlsx_bytes, "xlsx", strict=True)

        report.assert_not_called()


# ---------------------------------------------------------------------------
# CSV sniffer fallback
# ---------------------------------------------------------------------------


class TestCsvSnifferFallback:
    def test_tab_delimited_with_csv_format(self):
        """Tab-delimited content parsed as csv should use sniffer fallback."""
        content = "Name\tScore\nAlice\t90\nBob\t85"
        result = parse_file_content(content, "csv")
        assert result == [["Name", "Score"], ["Alice", "90"], ["Bob", "85"]]

    def test_sniffer_failure_returns_content(self):
        """When sniffer fails, single-column falls back to raw content."""
        content = "Name\nAlice\nBob"
        result = parse_file_content(content, "csv")
        assert result == content


# ---------------------------------------------------------------------------
# OpenpyxlInvalidFile fallback
# ---------------------------------------------------------------------------


class TestOpenpyxlFallback:
    def test_invalid_xlsx_non_strict(self):
        """Invalid xlsx bytes should fall back gracefully in non-strict mode."""
        result = parse_file_content(b"not xlsx bytes", "xlsx")
        assert result == b"not xlsx bytes"


# ---------------------------------------------------------------------------
# Header-only CSV
# ---------------------------------------------------------------------------


class TestHeaderOnlyCsv:
    def test_header_only_csv_returns_header_row(self):
        """CSV with only a header row (no data rows) should return [[header]]."""
        content = "Name,Score"
        result = parse_file_content(content, "csv")
        assert result == [["Name", "Score"]]

    def test_header_only_csv_with_trailing_newline(self):
        content = "Name,Score\n"
        result = parse_file_content(content, "csv")
        assert result == [["Name", "Score"]]


# ---------------------------------------------------------------------------
# Binary format + line range (line range ignored for binary formats)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_pyarrow, reason="pyarrow not installed")
class TestBinaryFormatLineRange:
    def test_parquet_ignores_line_range(self):
        """Binary formats should parse the full file regardless of line range.

        Line ranges are meaningless for binary formats (parquet/xlsx) — the
        caller (file_ref._expand_bare_ref) passes raw bytes and the parser
        should return the complete structured data.
        """
        import pandas as pd

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        # parse_file_content itself doesn't take a line range — this tests
        # that the full content is parsed even though the bytes could have
        # been truncated upstream (it's not, by design).
        result = parse_file_content(buf.getvalue(), "parquet")
        assert result == [["A", "B"], [1, 4], [2, 5], [3, 6]]


# ---------------------------------------------------------------------------
# Legacy .xls UX
# ---------------------------------------------------------------------------


class TestXlsFallback:
    def test_xls_returns_helpful_error_string(self):
        """Uploading a .xls file should produce a helpful error, not garbled binary."""
        result = parse_file_content(b"\xd0\xcf\x11\xe0garbled", "xls")
        assert isinstance(result, str)
        assert ".xlsx" in result
        assert "not supported" in result.lower()

    def test_xls_with_string_content(self):
        result = parse_file_content("some text", "xls")
        assert isinstance(result, str)
        assert ".xlsx" in result
