import io
from datetime import datetime

import pandas as pd
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


def _parquet(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()
