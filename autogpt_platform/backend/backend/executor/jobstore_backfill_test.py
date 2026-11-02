import os
import pickle
import tempfile
from contextlib import contextmanager
from enum import Enum
from unittest.mock import patch

import pytest
from sqlalchemy import (
    Column,
    Float,
    LargeBinary,
    MetaData,
    Table,
    Unicode,
    create_engine,
    select,
)
from sqlalchemy.exc import OperationalError

from backend.executor.jobstore_backfill import _has_enum, _normalize_table, _strip_enums

_BACKFILL = "backend.executor.jobstore_backfill"

TABLE = "apscheduler_jobs"


class Provider(str, Enum):
    GITHUB = "github"


class _Dropped(Enum):
    GONE = "gone"


# Pickled while GONE exists, then rebound without it — an unreadable row.
UNREADABLE = pickle.dumps({"kwargs": {"x": _Dropped.GONE}}, pickle.HIGHEST_PROTOCOL)


class _Dropped(Enum):  # noqa: F811 - deliberate rebind, drops GONE
    KEPT = "kept"


def test_str_enum_is_detected_despite_comparing_equal_to_its_value():
    """The bug this guards: Provider.GITHUB == "github" is True, so an
    equality check would call a poisoned row unchanged and skip it."""
    assert Provider.GITHUB == "github"
    assert _has_enum({"provider": Provider.GITHUB})
    assert not _has_enum({"provider": "github"})


def test_dry_run_reports_without_mutating():
    with _db() as engine:
        _insert(engine, "job1", {"provider": Provider.GITHUB})
        before = _job_state(engine, "job1")

        changed, unreadable, _ = _normalize_table(
            engine, MetaData(), TABLE, apply=False
        )

        assert (changed, unreadable) == (1, 0)
        assert _job_state(engine, "job1") == before


def test_apply_rewrites_the_enum_to_a_plain_value():
    with _db() as engine:
        _insert(engine, "job1", {"provider": Provider.GITHUB})

        changed, _, _ = _normalize_table(engine, MetaData(), TABLE, apply=True)

        assert changed == 1
        kwargs = pickle.loads(_job_state(engine, "job1"))["kwargs"]
        assert kwargs == {"provider": "github"}
        assert type(kwargs["provider"]) is str
        assert not isinstance(kwargs["provider"], Enum)


def test_apply_is_idempotent():
    with _db() as engine:
        _insert(engine, "job1", {"provider": Provider.GITHUB})
        _normalize_table(engine, MetaData(), TABLE, apply=True)
        after_first = _job_state(engine, "job1")

        changed, _, _ = _normalize_table(engine, MetaData(), TABLE, apply=True)

        assert changed == 0
        assert _job_state(engine, "job1") == after_first


def test_nested_and_sequence_enums_are_rewritten():
    with _db() as engine:
        _insert(
            engine,
            "job1",
            {"creds": {"a": {"provider": Provider.GITHUB}}, "seq": [Provider.GITHUB]},
        )

        _normalize_table(engine, MetaData(), TABLE, apply=True)

        kwargs = pickle.loads(_job_state(engine, "job1"))["kwargs"]
        assert kwargs == {"creds": {"a": {"provider": "github"}}, "seq": ["github"]}
        assert type(kwargs["creds"]["a"]["provider"]) is str
        assert type(kwargs["seq"][0]) is str


def test_unreadable_row_is_counted_and_left_untouched():
    """A backfill that mangles a row it cannot read is worse than one that skips."""
    with _db() as engine:
        _insert_raw(engine, "broken", UNREADABLE)
        _insert(engine, "job1", {"provider": Provider.GITHUB})

        changed, unreadable, _ = _normalize_table(engine, MetaData(), TABLE, apply=True)

        assert (changed, unreadable) == (1, 1)
        assert _job_state(engine, "broken") == UNREADABLE


def test_rows_without_enums_are_left_alone():
    with _db() as engine:
        _insert(engine, "job1", {"user_id": "u1", "cron": "0 * * * *"})
        before = _job_state(engine, "job1")

        changed, unreadable, _ = _normalize_table(engine, MetaData(), TABLE, apply=True)

        assert (changed, unreadable) == (0, 0)
        assert _job_state(engine, "job1") == before


def test_missing_table_is_skipped_not_raised():
    with _db() as engine:
        assert _normalize_table(engine, MetaData(), "no_such_table", apply=True) == (
            0,
            0,
            0,
        )


def test_a_reflection_failure_is_not_reported_as_a_missing_table():
    """A missing table is a skip; anything else must fail loudly. This script
    gates a deploy, so a connection or permission error exiting 0 would read as
    'nothing to rewrite'."""
    engine = create_engine("sqlite:////nonexistent-dir/nope.sqlite")

    with pytest.raises(OperationalError):
        _normalize_table(engine, MetaData(), TABLE, apply=True)


def test_a_row_that_changed_under_the_scan_is_left_alone():
    """The scheduler rewrites job_state whenever a job fires, and READ
    COMMITTED lets that land between the scan and the update."""
    with _db() as engine:
        _insert(engine, "job1", {"kwargs": {"provider": Provider.GITHUB}})
        newer = pickle.dumps({"kwargs": {"written": "by the scheduler"}})

        # Stand in for the concurrent writer: the value in the table is no
        # longer the one the scan read, so the predicate must not match.
        def _interfere(value):
            with engine.begin() as conn:
                conn.execute(
                    Table(TABLE, MetaData(), autoload_with=engine)
                    .update()
                    .where(Column("id", Unicode(191)) == "job1")
                    .values(job_state=newer)
                )
            return _strip_enums(value)

        with patch(f"{_BACKFILL}._strip_enums", side_effect=_interfere):
            changed, unreadable, skipped = _normalize_table(
                engine, MetaData(), TABLE, apply=True
            )

        assert (changed, skipped) == (0, 1)
        assert _job_state(engine, "job1") == newer


def test_repairing_a_parked_row_also_clears_its_pickled_next_run_time():
    """A parked row is paused in the COLUMN only, because parking could not
    deserialize it to rewrite the copy. Repairing it without clearing that copy
    strands the row — nothing reports it, nothing runs it, resume refuses it —
    so the repair has to finish the job."""
    with _db() as engine:
        _insert_raw(engine, "parked", _state(1234.0), next_run_time=None)

        changed, _, _ = _normalize_table(engine, MetaData(), TABLE, apply=True)

        assert changed == 1
        assert pickle.loads(_job_state(engine, "parked"))["next_run_time"] is None


def test_repairing_a_running_row_leaves_its_next_run_time_alone():
    with _db() as engine:
        _insert_raw(engine, "live", _state(1234.0), next_run_time=99.0)

        _normalize_table(engine, MetaData(), TABLE, apply=True)

        assert pickle.loads(_job_state(engine, "live"))["next_run_time"] == 1234.0


@contextmanager
def _db():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}")
    metadata = MetaData()
    Table(
        TABLE,
        metadata,
        Column("id", Unicode(191), primary_key=True),
        Column("next_run_time", Float(25), index=True),
        Column("job_state", LargeBinary, nullable=False),
    )
    metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()
        os.unlink(path)


def _state(next_run_time: float) -> bytes:
    return pickle.dumps(
        {
            "id": "x",
            "args": (),
            "kwargs": {"provider": Provider.GITHUB},
            "next_run_time": next_run_time,
        },
        pickle.HIGHEST_PROTOCOL,
    )


def _insert(engine, job_id: str, kwargs: dict) -> None:
    state = {"id": job_id, "args": (), "kwargs": kwargs}
    _insert_raw(engine, job_id, pickle.dumps(state, pickle.HIGHEST_PROTOCOL))


def _insert_raw(engine, job_id: str, job_state: bytes, next_run_time=1.0) -> None:
    table = Table(TABLE, MetaData(), autoload_with=engine)
    with engine.begin() as conn:
        conn.execute(
            table.insert().values(
                id=job_id, next_run_time=next_run_time, job_state=job_state
            )
        )


def _job_state(engine, job_id: str) -> bytes:
    table = Table(TABLE, MetaData(), autoload_with=engine)
    with engine.begin() as conn:
        return conn.execute(
            select(table.c.job_state).where(table.c.id == job_id)
        ).scalar()
