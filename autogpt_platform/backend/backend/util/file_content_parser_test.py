"""Tests for file_content_parser — format inference and structured parsing."""

import io
import json

import pytest

from backend.util.file_content_parser import (
    BINARY_FORMATS,
    infer_format,
    parse_file_content,
)

# ---------------------------------------------------------------------------
# infer_format
# ---------------------------------------------------------------------------


class TestInferFormat:
    # --- extension-based ---

    def test_json_extension(self):
        assert infer_format("/home/user/data.json") == "json"

    def test_jsonl_extension(self):
        assert infer_format("/tmp/events.jsonl") == "jsonl"

    def test_ndjson_extension(self):
        assert infer_format("/tmp/events.ndjson") == "jsonl"

    def test_csv_extension(self):
        assert infer_format("workspace:///reports/sales.csv") == "csv"

    def test_tsv_extension(self):
        assert infer_format("/home/user/data.tsv") == "tsv"

    def test_yaml_extension(self):
        assert infer_format("/home/user/config.yaml") == "yaml"

    def test_yml_extension(self):
        assert infer_format("/home/user/config.yml") == "yaml"

    def test_toml_extension(self):
        assert infer_format("/home/user/config.toml") == "toml"

    def test_parquet_extension(self):
        assert infer_format("/data/table.parquet") == "parquet"

    def test_xlsx_extension(self):
        assert infer_format("/data/spreadsheet.xlsx") == "xlsx"

    def test_xls_extension_not_supported(self):
        # Legacy .xls requires xlrd which we don't bundle.
        assert infer_format("/data/old_spreadsheet.xls") is None

    def test_case_insensitive(self):
        assert infer_format("/data/FILE.JSON") == "json"
        assert infer_format("/data/FILE.CSV") == "csv"

    def test_unknown_extension(self):
        assert infer_format("/home/user/readme.txt") is None

    def test_no_extension(self):
        assert infer_format("workspace://abc123") is None

    # --- MIME-based ---

    def test_mime_json(self):
        assert infer_format("workspace://abc123#application/json") == "json"

    def test_mime_csv(self):
        assert infer_format("workspace://abc123#text/csv") == "csv"

    def test_mime_tsv(self):
        assert infer_format("workspace://abc123#text/tab-separated-values") == "tsv"

    def test_mime_ndjson(self):
        assert infer_format("workspace://abc123#application/x-ndjson") == "jsonl"

    def test_mime_yaml(self):
        assert infer_format("workspace://abc123#application/x-yaml") == "yaml"

    def test_mime_xlsx(self):
        uri = "workspace://abc123#application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        assert infer_format(uri) == "xlsx"

    def test_mime_parquet(self):
        assert (
            infer_format("workspace://abc123#application/vnd.apache.parquet")
            == "parquet"
        )

    def test_unknown_mime(self):
        assert infer_format("workspace://abc123#text/plain") is None

    # --- MIME takes precedence over extension ---

    def test_mime_overrides_extension(self):
        # .txt extension but JSON MIME → json
        assert infer_format("workspace:///file.txt#application/json") == "json"


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
        # Either parses or falls back — should not raise.
        assert result is not None


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
