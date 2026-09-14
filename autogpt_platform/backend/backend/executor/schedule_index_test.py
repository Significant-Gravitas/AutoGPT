"""Unit tests for the schedule index (in-memory SQLite, no infra)."""

import sqlite3
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError

from backend.executor.schedule_index import ScheduleIndex, ScheduleIndexEntry


def _index() -> ScheduleIndex:
    connection = sqlite3.connect(":memory:")
    connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    index = ScheduleIndex(create_engine("sqlite://", creator=lambda: connection))
    index.ensure_table()
    return index


def _entry(job_id: str, **overrides) -> ScheduleIndexEntry:
    fields = {
        "job_id": job_id,
        "user_id": "user-1",
        "kind": "graph",
        "graph_id": "graph-1",
    }
    fields.update(overrides)
    return ScheduleIndexEntry(**fields)


# ---------------------------------------------------------------------------
# writes
# ---------------------------------------------------------------------------


def test_upsert_inserts_and_replaces_on_same_job_id():
    index = _index()
    index.upsert(_entry("j1", user_id="user-1"))
    index.upsert(_entry("j1", user_id="user-2"))

    assert index.candidate_job_ids(user_id="user-1") == []
    assert index.candidate_job_ids(user_id="user-2") == ["j1"]
    assert index.all_job_ids() == {"j1"}


def test_upsert_many_and_delete_many():
    index = _index()
    index.upsert_many([_entry("j1"), _entry("j2"), _entry("j3")])
    index.delete_many(["j1", "j3"])
    assert index.all_job_ids() == {"j2"}


def test_upsert_many_replaces_more_rows_than_the_bind_limit():
    index = _index()
    entries = [_entry(f"j{i}") for i in range(2001)]
    index.upsert_many(entries)
    index.upsert(_entry("untouched"))

    index.upsert_many(
        [entry.model_copy(update={"user_id": "user-2"}) for entry in entries]
    )

    assert index.candidate_job_ids(user_id="user-1") == ["untouched"]
    assert set(index.candidate_job_ids(user_id="user-2") or []) == {
        entry.job_id for entry in entries
    }


def test_delete_many_removes_more_rows_than_the_bind_limit():
    index = _index()
    job_ids = [f"j{i}" for i in range(2001)]
    for job_id in job_ids:
        index.upsert(_entry(job_id))
    index.upsert(_entry("untouched"))

    index.delete_many(job_ids)

    assert index.all_job_ids() == {"untouched"}


def test_upsert_many_rolls_back_all_batches_after_retry_exhaustion():
    index = _index()
    entries = [_entry(f"j{i}") for i in range(2001)]
    for entry in entries:
        index.upsert(entry)
    replacements = [entry.model_copy(update={"user_id": "user-2"}) for entry in entries]

    with patch.object(index._engine, "begin", wraps=index._engine.begin) as begin:
        with pytest.raises(IntegrityError):
            index.upsert_many([*replacements, replacements[0]])

    assert begin.call_count == 2
    assert index.candidate_job_ids(user_id="user-2") == []
    assert set(index.candidate_job_ids(user_id="user-1") or []) == {
        entry.job_id for entry in entries
    }


def test_upsert_many_retries_a_transient_integrity_error():
    index = _index()
    conflict = IntegrityError(None, None, sqlite3.IntegrityError("concurrent insert"))
    with patch.object(
        index._engine, "begin", side_effect=[conflict, index._engine.begin()]
    ):
        index.upsert_many([_entry("j1"), _entry("j2")])

    assert index.all_job_ids() == {"j1", "j2"}


def test_delete_missing_row_is_a_noop():
    index = _index()
    index.delete("nope")
    assert index.all_job_ids() == set()


def test_empty_batches_are_noops():
    index = _index()
    index.upsert_many([])
    index.delete_many([])
    assert index.all_job_ids() == set()


def test_empty_string_organization_is_stored_as_null():
    # Legacy GraphExecutionJobArgs default organization_id to "" — those
    # rows must never match an org-scoped query.
    index = _index()
    index.upsert(_entry("j1", organization_id=""))
    assert index.candidate_job_ids(organization_id="") == []
    assert index.candidate_job_ids(user_id="user-1") == ["j1"]


# ---------------------------------------------------------------------------
# candidate queries
# ---------------------------------------------------------------------------


def test_candidates_by_user():
    index = _index()
    index.upsert_many(
        [
            _entry("j1", user_id="a"),
            _entry("j2", user_id="b"),
            _entry("j3", user_id="a"),
        ]
    )
    assert sorted(index.candidate_job_ids(user_id="a") or []) == ["j1", "j3"]


def test_candidates_by_graph_and_session():
    index = _index()
    index.upsert_many(
        [
            _entry("j1", graph_id="g1"),
            _entry("j2", graph_id="g2"),
            _entry("j3", kind="copilot_turn", graph_id=None, session_id="s1"),
        ]
    )
    assert index.candidate_job_ids(graph_id="g1") == ["j1"]
    assert index.candidate_job_ids(session_id="s1") == ["j3"]


def test_kind_narrows_but_is_not_an_identity_filter():
    index = _index()
    index.upsert_many(
        [
            _entry("j1", user_id="a", kind="graph"),
            _entry("j2", user_id="a", kind="copilot_turn", graph_id=None),
        ]
    )
    assert index.candidate_job_ids(user_id="a", kind="graph") == ["j1"]
    # kind alone is a global listing: the index must decline (return None)
    # so the caller falls back to the full scan that includes un-indexed rows.
    assert index.candidate_job_ids(kind="graph") is None


def test_no_filters_returns_none_for_full_scan_fallback():
    index = _index()
    index.upsert(_entry("j1"))
    assert index.candidate_job_ids() is None


def test_org_scope_is_own_rows_or_org_rows():
    index = _index()
    index.upsert_many(
        [
            _entry("j1", user_id="me", organization_id=None),
            _entry("j2", user_id="teammate", organization_id="org-1"),
            _entry("j3", user_id="stranger", organization_id="org-2"),
            _entry("j4", user_id="teammate", organization_id=None),
        ]
    )
    got = index.candidate_job_ids(user_id="me", organization_id="org-1")
    # Superset semantics: own rows + all org rows; team/expert trimming is
    # the caller's predicate's job.
    assert sorted(got or []) == ["j1", "j2"]


def test_dimension_filters_are_anded():
    index = _index()
    index.upsert_many(
        [
            _entry("j1", user_id="a", graph_id="g1"),
            _entry("j2", user_id="a", graph_id="g2"),
            _entry("j3", user_id="b", graph_id="g1"),
        ]
    )
    assert index.candidate_job_ids(user_id="a", graph_id="g1") == ["j1"]
