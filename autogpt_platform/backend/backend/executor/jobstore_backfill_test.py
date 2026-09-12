import os
import pickle
import tempfile
from contextlib import contextmanager
from enum import Enum

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

from backend.executor.jobstore_backfill import _has_enum, _normalize_table

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

        changed, unreadable = _normalize_table(engine, MetaData(), TABLE, apply=False)

        assert (changed, unreadable) == (1, 0)
        assert _job_state(engine, "job1") == before


def test_apply_rewrites_the_enum_to_a_plain_value():
    with _db() as engine:
        _insert(engine, "job1", {"provider": Provider.GITHUB})

        changed, _ = _normalize_table(engine, MetaData(), TABLE, apply=True)

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

        changed, _ = _normalize_table(engine, MetaData(), TABLE, apply=True)

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

        changed, unreadable = _normalize_table(engine, MetaData(), TABLE, apply=True)

        assert (changed, unreadable) == (1, 1)
        assert _job_state(engine, "broken") == UNREADABLE


def test_rows_without_enums_are_left_alone():
    with _db() as engine:
        _insert(engine, "job1", {"user_id": "u1", "cron": "0 * * * *"})
        before = _job_state(engine, "job1")

        changed, unreadable = _normalize_table(engine, MetaData(), TABLE, apply=True)

        assert (changed, unreadable) == (0, 0)
        assert _job_state(engine, "job1") == before


def test_missing_table_is_skipped_not_raised():
    with _db() as engine:
        assert _normalize_table(engine, MetaData(), "no_such_table", apply=True) == (
            0,
            0,
        )


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


def _insert(engine, job_id: str, kwargs: dict) -> None:
    state = {"id": job_id, "args": (), "kwargs": kwargs}
    _insert_raw(engine, job_id, pickle.dumps(state, pickle.HIGHEST_PROTOCOL))


def _insert_raw(engine, job_id: str, job_state: bytes) -> None:
    table = Table(TABLE, MetaData(), autoload_with=engine)
    with engine.begin() as conn:
        conn.execute(
            table.insert().values(id=job_id, next_run_time=1.0, job_state=job_state)
        )


def _job_state(engine, job_id: str) -> bytes:
    table = Table(TABLE, MetaData(), autoload_with=engine)
    with engine.begin() as conn:
        return conn.execute(
            select(table.c.job_state).where(table.c.id == job_id)
        ).scalar()
