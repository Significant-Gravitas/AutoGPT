"""Parquet files read into rows in a short-lived child process.

Each read runs in a fresh interpreter started with ``python -m`` (never a fork of
the calling process) that refuses a file of more than ``MAX_PARQUET_CELLS`` cells
or values, caps its data segment at ``PARQUET_READ_MEMORY_BYTES`` past what its
imports hold, and is killed after ``PARQUET_READ_TIMEOUT_SECONDS``. It answers
with an exit code and at most ``MAX_PARQUET_JSON_BYTES`` of rows as JSON, never a
pickle, and receives only the environment variables in ``CHILD_ENV_ALLOWLIST``.
"""

import io
import json
import os
import subprocess
import sys
from typing import Any, cast

if sys.platform != "win32":
    import resource

MAX_PARQUET_CELLS = 1_000_000
PARQUET_READ_MEMORY_BYTES = 512 * 1024 * 1024
PARQUET_READ_TIMEOUT_SECONDS = 30
MAX_PARQUET_JSON_BYTES = 16 * 1024 * 1024

# What the interpreter needs to start and find the backend package.
CHILD_ENV_ALLOWLIST = (
    "PATH",
    "PYTHONPATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "SYSTEMROOT",
)

_CHILD_MODULE = "backend.util.parquet_rows"
_EXIT_OK = 0
_EXIT_TOO_LARGE = 3
_EXIT_UNREADABLE = 4
_MAX_ERROR_CHARS = 500


def read_parquet_rows(content: bytes) -> list[list[Any]]:
    """The file's rows after a header row of its column names. Missing values
    are None and dates are ISO 8601 strings.

    Raises ``ValueError`` for a file that is not Parquet, is over the limits,
    or takes too long to read.
    """
    try:
        done = subprocess.run(
            [sys.executable, "-m", _CHILD_MODULE],
            input=content,
            capture_output=True,
            timeout=PARQUET_READ_TIMEOUT_SECONDS,
            env=_child_env(),
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise ValueError(
            f"Parquet file took longer than {PARQUET_READ_TIMEOUT_SECONDS}s to read"
        ) from e
    if done.returncode == _EXIT_OK:
        table = json.loads(done.stdout)
        return [table["columns"], *table["data"]]
    if done.returncode == _EXIT_TOO_LARGE:
        raise ValueError(
            f"Parquet file is too large to read: over {MAX_PARQUET_CELLS:,} values, "
            f"or over {PARQUET_READ_MEMORY_BYTES // 2**20} MiB once read"
        )
    if done.returncode == _EXIT_UNREADABLE:
        reason = done.stderr.decode("utf-8", "replace")[:_MAX_ERROR_CHARS]
        raise ValueError(reason or "Not a readable Parquet file")
    raise ValueError("Parquet file could not be read")


def _child_env() -> dict[str, str]:
    return {
        name: os.environ[name] for name in CHILD_ENV_ALLOWLIST if name in os.environ
    }


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------


def _child_main() -> int:
    # Heavy, and needed only here: the calling process never imports them.
    import pandas as pd
    import pyarrow.parquet as pq

    content = sys.stdin.buffer.read()
    _limit_memory(_data_in_use())
    try:
        metadata = pq.ParquetFile(io.BytesIO(content)).metadata
        groups = [metadata.row_group(i) for i in range(metadata.num_row_groups)]
        rows = max(metadata.num_rows, sum(group.num_rows for group in groups))
        # A list cell holds many values, so count those as well as the cells.
        values = sum(
            group.column(c).num_values
            for group in groups
            for c in range(group.num_columns)
        )
        if max(rows * metadata.num_columns, values) > MAX_PARQUET_CELLS:
            return _EXIT_TOO_LARGE
        # With no path given, to_json returns the text.
        table = cast(
            str,
            pd.read_parquet(io.BytesIO(content)).to_json(
                orient="split",
                index=False,
                date_format="iso",
                double_precision=15,
                default_handler=str,
            ),
        )
    # The JSON encoder raises OverflowError when its buffer cannot grow.
    except (MemoryError, OverflowError):
        return _EXIT_TOO_LARGE
    except Exception as e:
        sys.stderr.write(str(e)[:_MAX_ERROR_CHARS])
        return _EXIT_UNREADABLE
    output = table.encode()
    if len(output) > MAX_PARQUET_JSON_BYTES:
        return _EXIT_TOO_LARGE
    sys.stdout.buffer.write(output)
    sys.stdout.buffer.flush()
    return _EXIT_OK


def _limit_memory(in_use: int) -> None:
    # Set only where it is enforced and VmData can be read.
    if sys.platform != "linux":
        return
    _, hard = resource.getrlimit(resource.RLIMIT_DATA)
    limit = in_use + PARQUET_READ_MEMORY_BYTES
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))


def _data_in_use() -> int:
    """Bytes of this process's data segment, which RLIMIT_DATA bounds."""
    if sys.platform != "linux":
        return 0
    with open("/proc/self/status") as status:
        return next(
            int(line.split()[1]) * 1024 for line in status if line.startswith("VmData:")
        )


if __name__ == "__main__":
    sys.exit(_child_main())
