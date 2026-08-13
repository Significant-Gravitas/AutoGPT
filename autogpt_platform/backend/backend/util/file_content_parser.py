"""Parse file content into structured Python objects based on file format.

Used by the ``@@agptfile:`` expansion system to eagerly parse well-known file
formats into native Python types *before* schema-driven coercion runs.  This
lets blocks with ``Any``-typed inputs receive structured data rather than raw
strings, while blocks expecting strings get the value coerced back via
``convert()``.

Supported formats:

- **JSON** (``.json``) — arrays and objects are promoted; scalars stay as strings
- **JSON Lines** (``.jsonl``, ``.ndjson``) — each non-empty line parsed as JSON;
  when all lines are dicts with the same keys (tabular data), output is
  ``list[list[Any]]`` with a header row, consistent with CSV/Parquet/Excel;
  otherwise returns a plain ``list`` of parsed values
- **CSV** (``.csv``) — ``csv.reader`` → ``list[list[str]]``
- **TSV** (``.tsv``) — tab-delimited → ``list[list[str]]``
- **YAML** (``.yaml``, ``.yml``) — parsed via PyYAML; containers only
- **TOML** (``.toml``) — parsed via stdlib ``tomllib``
- **Parquet** (``.parquet``) — via pandas/pyarrow → ``list[list[Any]]`` with header row
- **Excel** (``.xlsx``) — via pandas/openpyxl → ``list[list[Any]]`` with header row
  (legacy ``.xls`` is **not** supported — only the modern OOXML format)

The **fallback contract** is enforced by :func:`parse_file_content`, not by
individual parser functions.  If any parser raises, ``parse_file_content``
catches the exception and returns the original content unchanged (string for
text formats, bytes for binary formats).  Callers should never see an
exception from the public API when ``strict=False``.
"""

import csv
import io
import json
import logging
import tomllib
import zipfile
from collections.abc import Callable

# posixpath.splitext handles forward-slash URI paths correctly on all platforms,
# unlike os.path.splitext which uses platform-native separators.
from posixpath import splitext
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extension / MIME → format label mapping
# ---------------------------------------------------------------------------

_EXT_TO_FORMAT: dict[str, str] = {
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".csv": "csv",
    ".tsv": "tsv",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".parquet": "parquet",
    ".xlsx": "xlsx",
}

MIME_TO_FORMAT: dict[str, str] = {
    "application/json": "json",
    "application/x-ndjson": "jsonl",
    "application/jsonl": "jsonl",
    "text/csv": "csv",
    "text/tab-separated-values": "tsv",
    "application/x-yaml": "yaml",
    "application/yaml": "yaml",
    "text/yaml": "yaml",
    "application/toml": "toml",
    "application/vnd.apache.parquet": "parquet",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
}

# Formats that require raw bytes rather than decoded text.
BINARY_FORMATS: frozenset[str] = frozenset({"parquet", "xlsx"})


# ---------------------------------------------------------------------------
# Public API  (top-down: main functions first, helpers below)
# ---------------------------------------------------------------------------


def infer_format_from_uri(uri: str) -> str | None:
    """Return a format label based on URI extension or MIME fragment.

    Returns ``None`` when the format cannot be determined — the caller should
    fall back to returning the content as a plain string.
    """
    # 1. Check MIME fragment  (workspace://abc123#application/json)
    if "#" in uri:
        _, fragment = uri.rsplit("#", 1)
        fmt = MIME_TO_FORMAT.get(fragment.lower())
        if fmt:
            return fmt

    # 2. Check file extension from the path portion.
    #    Strip the fragment first so ".json#mime" doesn't confuse splitext.
    path = uri.split("#")[0].split("?")[0]
    _, ext = splitext(path)
    fmt = _EXT_TO_FORMAT.get(ext.lower())
    if fmt is not None:
        return fmt

    # Legacy .xls is not supported — map it so callers can produce a
    # user-friendly error instead of returning garbled binary.
    if ext.lower() == ".xls":
        return "xls"

    return None


def parse_file_content(content: str | bytes, fmt: str, *, strict: bool = False) -> Any:
    """Parse *content* according to *fmt* and return a native Python value.

    When *strict* is ``False`` (default), returns the original *content*
    unchanged if *fmt* is not recognised or parsing fails for any reason.
    This mode **never raises**.

    When *strict* is ``True``, parsing errors are propagated to the caller.
    Unrecognised formats or type mismatches (e.g. text for a binary format)
    still return *content* unchanged without raising.
    """
    if fmt == "xls":
        return (
            "[Unsupported format] Legacy .xls files are not supported. "
            "Please re-save the file as .xlsx (Excel 2007+) and upload again."
        )

    try:
        if fmt in BINARY_FORMATS:
            parser = _BINARY_PARSERS.get(fmt)
            if parser is None:
                return content
            if isinstance(content, str):
                # Caller gave us text for a binary format — can't parse.
                return content
            return parser(content)

        parser = _TEXT_PARSERS.get(fmt)
        if parser is None:
            return content
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        return parser(content)

    except PARSE_EXCEPTIONS:
        if strict:
            raise
        logger.debug("Structured parsing failed for format=%s, falling back", fmt)
        return content


# ---------------------------------------------------------------------------
# Exception loading helpers
# ---------------------------------------------------------------------------


def _load_openpyxl_exception() -> type[Exception]:
    """Return openpyxl's InvalidFileException, raising ImportError if absent."""
    from openpyxl.utils.exceptions import InvalidFileException  # noqa: PLC0415

    return InvalidFileException


def _load_arrow_exception() -> type[Exception]:
    """Return pyarrow's ArrowException, raising ImportError if absent."""
    from pyarrow import ArrowException  # noqa: PLC0415

    return ArrowException


def _optional_exc(loader: "Callable[[], type[Exception]]") -> "type[Exception] | None":
    """Return the exception class from *loader*, or ``None`` if the dep is absent."""
    try:
        return loader()
    except ImportError:
        return None


# Exception types that can be raised during file content parsing.
# Shared between ``parse_file_content`` (which catches them in non-strict mode)
# and ``file_ref._expand_bare_ref`` (which re-raises them as FileRefExpansionError).
#
# Optional-dependency exception types are loaded via a helper that raises
# ``ImportError`` at *parse time* rather than silently becoming ``None`` here.
# This ensures mypy sees clean types and missing deps surface as real errors.
PARSE_EXCEPTIONS: tuple[type[BaseException], ...] = tuple(
    exc
    for exc in (
        json.JSONDecodeError,
        csv.Error,
        yaml.YAMLError,
        tomllib.TOMLDecodeError,
        ValueError,
        UnicodeDecodeError,
        ImportError,
        OSError,
        KeyError,
        TypeError,
        zipfile.BadZipFile,
        _optional_exc(_load_openpyxl_exception),
        # ArrowException covers ArrowIOError and ArrowCapacityError which
        # do not inherit from standard exceptions; ArrowInvalid/ArrowTypeError
        # already map to ValueError/TypeError but this catches the rest.
        _optional_exc(_load_arrow_exception),
    )
    if exc is not None
)


# ---------------------------------------------------------------------------
# Text-based parsers  (content: str → Any)
# ---------------------------------------------------------------------------


def _parse_container(parser: Callable[[str], Any], content: str) -> list | dict | str:
    """Parse *content* and return the result only if it is a container (list/dict).

    Scalar values (strings, numbers, booleans, None) are discarded and the
    original *content* string is returned instead.  This prevents e.g. a JSON
    file containing just ``"42"`` from silently becoming an int.
    """
    parsed = parser(content)
    if isinstance(parsed, (list, dict)):
        return parsed
    return content


def _parse_json(content: str) -> list | dict | str:
    return _parse_container(json.loads, content)


def _parse_jsonl(content: str) -> Any:
    lines = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not lines:
        return content

    # When every line is a dict with the same keys, convert to table format
    # (header row + data rows) — consistent with CSV/TSV/Parquet/Excel output.
    # Require ≥2 dicts so a single-line JSONL stays as [dict] (not a table).
    if len(lines) >= 2 and all(isinstance(obj, dict) for obj in lines):
        keys = list(lines[0].keys())
        # Cache as tuple to avoid O(n×k) list allocations in the all() call.
        keys_tuple = tuple(keys)
        if keys and all(tuple(obj.keys()) == keys_tuple for obj in lines[1:]):
            return [keys] + [[obj[k] for k in keys] for obj in lines]

    return lines


def _parse_csv(content: str) -> Any:
    return _parse_delimited(content, delimiter=",")


def _parse_tsv(content: str) -> Any:
    return _parse_delimited(content, delimiter="\t")


def _parse_delimited(content: str, *, delimiter: str) -> Any:
    reader = csv.reader(io.StringIO(content), delimiter=delimiter)
    # csv.reader never yields [] — blank lines yield [""]. Filter out
    # rows where every cell is empty (i.e. truly blank lines).
    rows = [row for row in reader if _row_has_content(row)]
    if not rows:
        return content
    # If the declared delimiter produces only single-column rows, try
    # sniffing the actual delimiter — catches misidentified files (e.g.
    # a tab-delimited file with a .csv extension).
    if len(rows[0]) == 1:
        try:
            dialect = csv.Sniffer().sniff(content[:8192])
            if dialect.delimiter != delimiter:
                reader = csv.reader(io.StringIO(content), dialect)
                rows = [row for row in reader if _row_has_content(row)]
        except csv.Error:
            pass
    if rows and len(rows[0]) >= 2:
        return rows
    return content


def _row_has_content(row: list[str]) -> bool:
    """Return True when *row* contains at least one non-empty cell.

    ``csv.reader`` never yields ``[]`` — truly blank lines yield ``[""]``.
    This predicate filters those out consistently across the initial read
    and the sniffer-fallback re-read.
    """
    return any(cell for cell in row)


def _parse_yaml(content: str) -> list | dict | str:
    # NOTE: YAML anchor/alias expansion can amplify input beyond the 10MB cap.
    # safe_load prevents code execution; for production hardening consider
    # a YAML parser with expansion limits (e.g. ruamel.yaml with max_alias_count).
    if "\n---" in content or content.startswith("---\n"):
        # Multi-document YAML: only the first document is parsed; the rest
        # are silently ignored by yaml.safe_load.  Warn so callers are aware.
        logger.warning(
            "Multi-document YAML detected (--- separator); "
            "only the first document will be parsed."
        )
    return _parse_container(yaml.safe_load, content)


def _parse_toml(content: str) -> Any:
    parsed = tomllib.loads(content)
    # tomllib.loads always returns a dict — return it even if empty.
    return parsed


_TEXT_PARSERS: dict[str, Callable[[str], Any]] = {
    "json": _parse_json,
    "jsonl": _parse_jsonl,
    "csv": _parse_csv,
    "tsv": _parse_tsv,
    "yaml": _parse_yaml,
    "toml": _parse_toml,
}

# ---------------------------------------------------------------------------
# Binary-based parsers  (content: bytes → Any)
# ---------------------------------------------------------------------------


def _parse_parquet(content: bytes) -> list[list[Any]]:
    import pandas as pd

    df = pd.read_parquet(io.BytesIO(content))
    return _df_to_rows(df)


def _parse_xlsx(content: bytes) -> list[list[Any]]:
    import pandas as pd

    # Explicitly specify openpyxl engine; the default engine varies by pandas
    # version and does not support legacy .xls (which is excluded by our format map).
    df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
    return _df_to_rows(df)


def _df_to_rows(df: Any) -> list[list[Any]]:
    """Convert a DataFrame to ``list[list[Any]]`` with a header row.

    NaN values are replaced with ``None`` so the result is JSON-serializable.
    Uses explicit cell-level checking because ``df.where(df.notna(), None)``
    silently converts ``None`` back to ``NaN`` in float64 columns.
    """
    header = df.columns.tolist()
    rows = [
        [None if _is_nan(cell) else cell for cell in row] for row in df.values.tolist()
    ]
    return [header] + rows


def _is_nan(cell: Any) -> bool:
    """Check if a cell value is NaN, handling non-scalar types (lists, dicts).

    ``pd.isna()`` on a list/dict returns a boolean array which raises
    ``ValueError`` in a boolean context.  Guard with a scalar check first.
    """
    import pandas as pd

    return bool(pd.api.types.is_scalar(cell) and pd.isna(cell))


_BINARY_PARSERS: dict[str, Callable[[bytes], Any]] = {
    "parquet": _parse_parquet,
    "xlsx": _parse_xlsx,
}
