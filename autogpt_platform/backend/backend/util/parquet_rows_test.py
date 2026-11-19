import io
import json
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.util import parquet_rows
from backend.util.parquet_rows import MAX_PARQUET_CELLS, read_parquet_rows


def test_a_table_comes_back_as_rows_with_iso_dates():
    rows = read_parquet_rows(
        _parquet(
            pd.DataFrame(
                {
                    "name": ["Ada", None],
                    "score": [1.5, float("nan")],
                    "seen": [datetime(2026, 1, 2, 3, 4, 5), datetime(2026, 1, 3)],
                    "tags": [["a", "b"], []],
                }
            )
        )
    )

    assert rows[0] == ["name", "score", "seen", "tags"]
    assert rows[1][:2] == ["Ada", 1.5]
    assert rows[1][2].startswith("2026-01-02T03:04:05")
    assert rows[1][3] == ["a", "b"]
    assert rows[2][:2] == [None, None]


def test_a_table_over_the_cell_limit_is_refused():
    with pytest.raises(ValueError, match="too large"):
        read_parquet_rows(_parquet(pd.DataFrame({"a": [0] * (MAX_PARQUET_CELLS + 1)})))


def test_a_read_past_its_time_limit_is_refused(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(parquet_rows, "PARQUET_READ_TIMEOUT_SECONDS", 0.001)

    with pytest.raises(ValueError, match="longer than"):
        read_parquet_rows(_parquet(pd.DataFrame({"a": [1]})))


def test_the_child_gets_only_the_allowed_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    seen: dict = {}
    real_run = subprocess.run

    def run(*args, **kwargs):
        seen.update(kwargs["env"])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(parquet_rows.subprocess, "run", run)

    read_parquet_rows(_parquet(pd.DataFrame({"a": [1]})))

    assert "DATABASE_URL" not in seen
    assert set(seen) <= set(parquet_rows.CHILD_ENV_ALLOWLIST)


def test_the_memory_cap_holds_in_a_fresh_process():
    allocate = (
        "from backend.util import parquet_rows as p\n"
        "p._limit_memory(p._data_in_use())\n"
        "bytearray(p.PARQUET_READ_MEMORY_BYTES + 64 * 2**20)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", allocate], capture_output=True, timeout=60
    )

    assert done.returncode != 0
    assert b"MemoryError" in done.stderr


@pytest.mark.parametrize(
    "case, code",
    [
        ("table", parquet_rows._EXIT_OK),
        ("over the cell limit", parquet_rows._EXIT_TOO_LARGE),
        ("one list cell over the limit", parquet_rows._EXIT_TOO_LARGE),
        ("not parquet", parquet_rows._EXIT_UNREADABLE),
    ],
)
def test_the_child_answers_with_its_exit_code(
    monkeypatch: pytest.MonkeyPatch, case: str, code: int
):
    content = {
        "table": lambda: _parquet(pd.DataFrame({"a": [1, 2]})),
        "over the cell limit": lambda: _parquet(
            pd.DataFrame({"a": [0] * (MAX_PARQUET_CELLS + 1)})
        ),
        "one list cell over the limit": lambda: _list_cell(MAX_PARQUET_CELLS + 1),
        "not parquet": lambda: b"not parquet",
    }[case]()
    limits, stdout = _child_in_process(monkeypatch, content)

    assert parquet_rows._child_main() == code
    # The cap is set before the file is looked at.
    assert len(limits) == 1
    if code == parquet_rows._EXIT_OK:
        assert json.loads(stdout.getvalue()) == {"columns": ["a"], "data": [[1], [2]]}


def test_the_child_refuses_rows_over_its_json_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(parquet_rows, "MAX_PARQUET_JSON_BYTES", 10)
    _, stdout = _child_in_process(
        monkeypatch, _parquet(pd.DataFrame({"a": list(range(100))}))
    )

    assert parquet_rows._child_main() == parquet_rows._EXIT_TOO_LARGE
    assert stdout.getvalue() == b""


def _child_in_process(
    monkeypatch: pytest.MonkeyPatch, content: bytes
) -> tuple[list[int], io.BytesIO]:
    """Run the child's code here, where coverage sees it, without setting an OS
    memory limit on the test process."""
    limits: list[int] = []
    stdout = io.BytesIO()
    monkeypatch.setattr(parquet_rows, "_limit_memory", limits.append)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(content)))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=stdout))
    return limits, stdout


def _list_cell(length: int) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.table({"a": [[0] * length]}), buf)
    return buf.getvalue()


def _parquet(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()
