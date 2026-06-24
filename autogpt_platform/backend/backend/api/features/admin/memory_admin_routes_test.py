"""Tests for the admin memory inspector routes."""

from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from redis.exceptions import ResponseError

from .memory_admin_routes import router as memory_admin_router

app = fastapi.FastAPI()
app.include_router(memory_admin_router)

client = fastapi.testclient.TestClient(app)

_MOCK_MODULE = "backend.api.features.admin.memory_admin_routes"


@pytest.fixture(autouse=True)
def setup_app_admin_auth(mock_jwt_admin):
    """Admin auth override for all tests in this module."""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _driver_returning(*query_results) -> AsyncMock:
    """Build a mocked AutoGPTFalkorDriver whose ``execute_query`` returns
    each value in turn (one per call)."""
    driver = AsyncMock()
    driver.execute_query.side_effect = [(r, None, None) for r in query_results]
    driver.close = AsyncMock()
    return driver


class TestOverview:
    def test_returns_counts_for_all_node_types(self) -> None:
        # Returns one row per count query in order: entities, episodes,
        # relates_to, mentions, communities.
        driver = _driver_returning(
            [{"c": 107}],  # entities
            [{"c": 36}],  # episodes
            [{"c": 104}],  # relates_to
            [{"c": 153}],  # mentions
            [{"c": 30}],  # communities
        )
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            resp = client.get("/admin/memory/abc/overview")
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == "abc"
        assert body["group_id"] == "user_abc"
        assert body["entities"] == 107
        assert body["episodes"] == 36
        assert body["relates_to_edges"] == 104
        assert body["mentions_edges"] == 153
        assert body["communities"] == 30
        # Driver must be closed even on success
        driver.close.assert_awaited_once()

    def test_me_resolves_to_caller_id(self, mock_jwt_admin) -> None:
        driver = _driver_returning(
            [{"c": 0}], [{"c": 0}], [{"c": 0}], [{"c": 0}], [{"c": 0}]
        )
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            resp = client.get("/admin/memory/me/overview")
        assert resp.status_code == 200
        body = resp.json()
        # 'me' resolved to the caller's admin_user_id from mock_jwt_admin
        assert body["user_id"] == mock_jwt_admin["user_id"]

    def test_invalid_user_id_returns_400(self) -> None:
        resp = client.get("/admin/memory/!!!invalid!!!/overview")
        assert resp.status_code == 400

    def test_missing_graph_returns_zeros(self) -> None:
        # Simulate the FalkorDB error for a database that doesn't exist yet.
        driver = AsyncMock()
        driver.execute_query.side_effect = ResponseError(
            "Invalid graph operation on empty key"
        )
        driver.close = AsyncMock()
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            resp = client.get("/admin/memory/abc/overview")
        assert resp.status_code == 200
        body = resp.json()
        assert body["entities"] == 0
        assert body["communities"] == 0

    def test_unexpected_cypher_error_propagates(self) -> None:
        """A Cypher typo / unrelated FalkorDB error must NOT be swallowed
        as a zero count — admins need to see the failure."""
        driver = AsyncMock()
        driver.execute_query.side_effect = ResponseError("syntax error near 'MATHC'")
        driver.close = AsyncMock()
        # Surface the underlying exception instead of letting the test
        # client convert it to a 500 — we want to assert the exact type
        # propagated past _count.
        bare_client = fastapi.testclient.TestClient(app, raise_server_exceptions=True)
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            with pytest.raises(ResponseError, match="MATHC"):
                bare_client.get("/admin/memory/abc/overview")
        # Driver must still be closed even when the query blows up.
        driver.close.assert_awaited_once()

    def test_non_response_error_propagates(self) -> None:
        """Non-FalkorDB exceptions (e.g. a programming TypeError) must not
        be silently zeroed."""
        driver = AsyncMock()
        driver.execute_query.side_effect = TypeError("bad query param")
        driver.close = AsyncMock()
        bare_client = fastapi.testclient.TestClient(app, raise_server_exceptions=True)
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            with pytest.raises(TypeError, match="bad query param"):
                bare_client.get("/admin/memory/abc/overview")
        driver.close.assert_awaited_once()


class TestListEntities:
    def test_returns_entity_summaries(self) -> None:
        driver = _driver_returning(
            [
                {"uuid": "e1", "name": "Alice", "summary": "engineer"},
                {"uuid": "e2", "name": "Atlas", "summary": None},
            ]
        )
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            resp = client.get("/admin/memory/abc/entities?limit=10")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 2
        assert items[0]["name"] == "Alice"
        assert items[1]["summary"] is None

    def test_limit_above_cap_rejected_with_422(self) -> None:
        """The entities route caps ``limit`` at 10000 via FastAPI Query
        validation. Anything beyond gets a 422 — not a 500 from a runaway
        Cypher LIMIT clause."""
        resp = client.get("/admin/memory/abc/entities?limit=10001")
        assert resp.status_code == 422


class TestListFacts:
    def test_default_returns_any_status(self) -> None:
        driver = _driver_returning(
            [
                {
                    "uuid": "e1",
                    "source": "Alice",
                    "target": "Atlas",
                    "name": "works_on",
                    "fact": "Alice works on Atlas",
                    "status": "active",
                    "scope": "real:global",
                    "confidence": 0.9,
                    "created_at": "2026-05-19T10:00:00Z",
                    "expired_at": None,
                }
            ]
        )
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            resp = client.get("/admin/memory/abc/facts")
        assert resp.status_code == 200
        # No status filter applied → query shouldn't include WHERE on status
        call_query = driver.execute_query.call_args.args[0]
        assert "e.status = $status" not in call_query
        assert resp.json()["items"][0]["source"] == "Alice"

    def test_status_filter_passed_to_cypher(self) -> None:
        driver = _driver_returning([])
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            resp = client.get(
                "/admin/memory/abc/facts?status=superseded&scope=project:atlas"
            )
        assert resp.status_code == 200
        kwargs = driver.execute_query.call_args.kwargs
        assert kwargs["status"] == "superseded"
        assert kwargs["scope"] == "project:atlas"

    def test_status_must_be_valid_enum(self) -> None:
        resp = client.get("/admin/memory/abc/facts?status=garbage")
        assert resp.status_code == 422


class TestListCommunities:
    def test_returns_community_summaries(self) -> None:
        driver = _driver_returning(
            [
                {
                    "uuid": "c1",
                    "name": "Card art",
                    "summary": "MTG card illustrations",
                    "member_count": 7,
                },
                {
                    "uuid": "c2",
                    "name": "Engineering",
                    "summary": "Software work",
                    "member_count": 4,
                },
            ]
        )
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            resp = client.get("/admin/memory/abc/communities?limit=10")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 2
        # Member counts come through correctly
        assert items[0]["member_count"] == 7


class TestGraph:
    def test_missing_graph_returns_empty(self) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = ResponseError("no such graph")
        driver.close = AsyncMock()
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            resp = client.get("/admin/memory/abc/graph")
        assert resp.status_code == 200
        body = resp.json()
        assert body["nodes"] == []
        assert body["edges"] == []

    def test_unexpected_error_propagates(self) -> None:
        """A non-graph-not-found FalkorDB error must surface, not be
        hidden behind an empty graph payload."""
        driver = AsyncMock()
        driver.execute_query.side_effect = ResponseError("syntax error near 'MATHC'")
        driver.close = AsyncMock()
        bare_client = fastapi.testclient.TestClient(app, raise_server_exceptions=True)
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            with pytest.raises(ResponseError, match="MATHC"):
                bare_client.get("/admin/memory/abc/graph")
        driver.close.assert_awaited_once()


class TestRebuildCommunitiesPolling:
    """POST /communities/rebuild is now fire-and-forget.

    It writes a JobStatus row, kicks off the scheduler via
    ``schedule_immediate_community_rebuild``, and returns 202 +
    job_id immediately. Frontend polls the GET endpoint to read
    progress; the work body fires asynchronously on the scheduler
    thread pool.
    """

    def test_returns_202_with_job_id(self) -> None:
        scheduler = MagicMock()
        scheduler.schedule_immediate_community_rebuild = AsyncMock(
            return_value={"scheduled": True, "job_id": "x", "kind": "rebuild"}
        )
        with patch(
            f"{_MOCK_MODULE}.get_scheduler_client", return_value=scheduler
        ), patch(
            f"{_MOCK_MODULE}.write_initial_status",
            new=_make_fake_initial_status("rebuild"),
        ):
            resp = client.post("/admin/memory/abc/communities/rebuild")
        assert resp.status_code == 202
        body = resp.json()
        assert body["kind"] == "rebuild"
        assert body["state"] == "queued"
        assert body["user_id"] == "abc"
        assert isinstance(body["job_id"], str)
        scheduler.schedule_immediate_community_rebuild.assert_awaited_once()
        # The job_id passed to the scheduler must match the one returned.
        call_kwargs = scheduler.schedule_immediate_community_rebuild.call_args.kwargs
        assert call_kwargs["job_id"] == body["job_id"]
        assert call_kwargs["user_id"] == "abc"

    def test_scheduler_failure_marks_errored_and_returns_500(self) -> None:
        scheduler = MagicMock()
        scheduler.schedule_immediate_community_rebuild = AsyncMock(
            side_effect=RuntimeError("scheduler unreachable")
        )
        mark_errored = AsyncMock()
        with patch(
            f"{_MOCK_MODULE}.get_scheduler_client", return_value=scheduler
        ), patch(
            f"{_MOCK_MODULE}.write_initial_status",
            new=_make_fake_initial_status("rebuild"),
        ), patch(
            f"{_MOCK_MODULE}.mark_errored", mark_errored
        ):
            resp = client.post("/admin/memory/abc/communities/rebuild")
        assert resp.status_code == 500
        assert "scheduler unreachable" in resp.json()["detail"]
        # The just-queued JobStatus must be flipped to errored, not left
        # stuck on 'queued' for a job the scheduler never picked up.
        mark_errored.assert_awaited_once()
        assert mark_errored.call_args.kwargs["kind"] == "rebuild"


class TestTriggerStatusGetEndpoints:
    """Each trigger has a paired GET that reads the JobStatus row."""

    def test_status_returns_404_for_unknown_job(self) -> None:
        with patch(f"{_MOCK_MODULE}.read_status", new=AsyncMock(return_value=None)):
            resp = client.get("/admin/memory/abc/nightly/missing-job-id")
        assert resp.status_code == 404

    def test_status_returns_403_when_user_mismatches(self) -> None:
        # A job owned by a different user must not be visible.
        status = _fake_status_row("nightly", "owned-by-someone-else")
        with patch(f"{_MOCK_MODULE}.read_status", new=AsyncMock(return_value=status)):
            resp = client.get("/admin/memory/abc/nightly/job-1")
        assert resp.status_code == 403

    def test_status_returns_job_row_for_owner(self) -> None:
        status = _fake_status_row("nightly", "abc")
        with patch(f"{_MOCK_MODULE}.read_status", new=AsyncMock(return_value=status)):
            resp = client.get("/admin/memory/abc/nightly/job-1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == "abc"
        assert body["kind"] == "nightly"
        assert body["state"] == "queued"

    def test_each_kind_has_its_own_status_endpoint(self) -> None:
        """Pin the URL structure: nightly, dream, communities/rebuild
        each have a GET .../{job_id} sibling to their POST."""
        for kind, url in [
            ("nightly", "/admin/memory/abc/nightly/job-1"),
            ("dream_pass", "/admin/memory/abc/dream/job-1"),
            ("rebuild", "/admin/memory/abc/communities/rebuild/job-1"),
        ]:
            status = _fake_status_row(kind, "abc")
            with patch(
                f"{_MOCK_MODULE}.read_status",
                new=AsyncMock(return_value=status),
            ):
                resp = client.get(url)
            assert resp.status_code == 200, url
            assert resp.json()["kind"] == kind


def _make_fake_initial_status(kind: str):
    """A ``write_initial_status`` replacement that returns a fake
    ``JobStatus`` matching the requested kind + caller's user_id."""

    async def fake(*, kind: str, job_id: str, user_id: str):
        return _fake_status_row(kind, user_id, job_id=job_id)

    return fake


def _fake_status_row(kind: str, user_id: str, job_id: str = "job-1"):
    from datetime import datetime, timezone

    from backend.copilot.dream.job_status import JobStatus

    now = datetime.now(timezone.utc)
    return JobStatus(
        job_id=job_id,
        user_id=user_id,
        kind=kind,
        state="queued",
        started_at=now,
        updated_at=now,
    )


class TestAdminGating:
    """Non-admin callers must get 403 on every route."""

    def test_non_admin_gets_403(self, mock_jwt_user) -> None:
        # Swap to the non-admin user fixture; the autouse setup_app_admin_auth
        # fixture restores admin auth (clears overrides) on teardown, so no
        # manual reset is needed.
        app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
        for path in [
            "/admin/memory/abc/overview",
            "/admin/memory/abc/entities",
            "/admin/memory/abc/facts",
            "/admin/memory/abc/communities",
        ]:
            resp = client.get(path)
            assert resp.status_code == 403, path
        resp = client.post("/admin/memory/abc/communities/rebuild")
        assert resp.status_code == 403
