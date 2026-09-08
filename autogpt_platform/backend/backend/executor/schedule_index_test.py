"""Unit tests for the schedule index (in-memory SQLite, no infra)."""

from sqlalchemy import create_engine

from backend.executor.schedule_index import ScheduleIndex, ScheduleIndexEntry


def _index() -> ScheduleIndex:
    index = ScheduleIndex(create_engine("sqlite://"))
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
