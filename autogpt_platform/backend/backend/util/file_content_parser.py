"""Parse file content into structured Python objects based on file format.

Used by the ``@@agptfile:`` expansion system to eagerly parse well-known file
formats into native Python types *before* schema-driven coercion runs.  This
lets blocks with ``Any``-typed inputs receive structured data rather than raw
strings, while blocks expecting strings get the value coerced back via
``convert()``.

Supported formats:

- **JSON** (``.json``) — arrays and objects are promoted; scalars stay as strings
- **JSON Lines** (``.jsonl``, ``.ndjson``) — each non-empty line parsed as JSON → list
- **CSV** (``.csv``) — ``csv.reader`` → ``list[list[str]]``
- **TSV** (``.tsv``) — tab-delimited → ``list[list[str]]``
- **YAML** (``.yaml``, ``.yml``) — parsed via PyYAML; containers only
- **TOML** (``.toml``) — parsed via stdlib ``tomllib``
- **Parquet** (``.parquet``) — via pandas/pyarrow → ``list[list[Any]]`` with header row
- **Excel** (``.xlsx``, ``.xls``) — via pandas → ``list[list[Any]]`` with header row

All parsers follow the **fallback contract**: if parsing fails for *any* reason,
the original content is returned unchanged (string for text formats, bytes for
binary formats).  Callers should never see an exception from this module.
"""

import csv
import io
import json
import logging
import tomllib
from posixpath import splitext
from typing import Any

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
    ".xls": "xlsx",
}

_MIME_TO_FORMAT: dict[str, str] = {
    "application/json": "json",
    "application/x-ndjson": "jsonl",
    "application/jsonl": "jsonl",
    "text/csv": "csv",
    "text/tab-separated-values": "tsv",
    "application/x-yaml": "yaml",
    "text/yaml": "yaml",
    "application/toml": "toml",
    "application/vnd.apache.parquet": "parquet",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xlsx",
}

# Formats that require raw bytes rather than decoded text.
BINARY_FORMATS: frozenset[str] = frozenset({"parquet", "xlsx"})


def infer_format(uri: str) -> str | None:
    """Return a format label based on URI extension or MIME fragment.

    Returns ``None`` when the format cannot be determined — the caller should
    fall back to returning the content as a plain string.
    """
    # 1. Check MIME fragment  (workspace://abc123#application/json)
    if "#" in uri:
        _, fragment = uri.rsplit("#", 1)
        fmt = _MIME_TO_FORMAT.get(fragment)
        if fmt:
            return fmt

    # 2. Check file extension from the path portion.
    #    Strip the fragment first so ".json#mime" doesn't confuse splitext.
    path = uri.split("#")[0]
    _, ext = splitext(path)
    return _EXT_TO_FORMAT.get(ext.lower())


# ---------------------------------------------------------------------------
# Text-based parsers  (content: str → Any)
# ---------------------------------------------------------------------------


def _parse_json(content: str) -> Any:
    parsed = json.loads(content)
    # Only promote containers.  Scalar JSON values (strings, numbers,
    # booleans, null) stay as the raw string so that e.g. a file containing
    # just ``"42"`` doesn't silently become an int.
    if isinstance(parsed, (list, dict)):
        return parsed
    return content


def _parse_jsonl(content: str) -> Any:
    lines = [json.loads(line) for line in content.splitlines() if line.strip()]
    return lines if lines else content


def _parse_csv(content: str) -> Any:
    return _parse_delimited(content, delimiter=",")


def _parse_tsv(content: str) -> Any:
    return _parse_delimited(content, delimiter="\t")


def _parse_delimited(content: str, *, delimiter: str) -> Any:
    reader = csv.reader(io.StringIO(content), delimiter=delimiter)
    rows = [row for row in reader if row]
    # Require ≥1 row and ≥2 columns to qualify as tabular data.
    if rows and len(rows[0]) >= 2:
        return rows
    return content


def _parse_yaml(content: str) -> Any:
    import yaml

    parsed = yaml.safe_load(content)
    if isinstance(parsed, (list, dict)):
        return parsed
    return content


def _parse_toml(content: str) -> Any:
    parsed = tomllib.loads(content)
    # tomllib.loads always returns a dict.
    return parsed if parsed else content


_TEXT_PARSERS: dict[str, Any] = {
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


def _parse_parquet(content: bytes) -> Any:
    import pandas as pd

    df = pd.read_parquet(io.BytesIO(content))
    # Return as list[list[Any]] with the first row being the header.
    header = df.columns.tolist()
    rows = df.values.tolist()
    return [header] + rows


def _parse_xlsx(content: bytes) -> Any:
    import pandas as pd

    df = pd.read_excel(io.BytesIO(content))
    header = df.columns.tolist()
    rows = df.values.tolist()
    return [header] + rows


_BINARY_PARSERS: dict[str, Any] = {
    "parquet": _parse_parquet,
    "xlsx": _parse_xlsx,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_file_content(content: str | bytes, fmt: str) -> Any:
    """Parse *content* according to *fmt* and return a native Python value.

    Returns the original *content* unchanged if:
    - *fmt* is not recognised
    - parsing fails for any reason (malformed content, missing dependency, etc.)

    This function **never raises**.
    """
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

    except Exception:
        logger.debug("Structured parsing failed for format=%s, falling back", fmt)
        return content
