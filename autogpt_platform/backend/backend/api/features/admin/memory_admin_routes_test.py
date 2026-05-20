"""Tests for the admin memory inspector routes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload

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

    def test_me_resolves_to_caller_id(
        self, mock_jwt_admin
    ) -> None:
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
        # Simulate a FalkorDB error on every query (database doesn't exist)
        driver = AsyncMock()
        driver.execute_query.side_effect = RuntimeError("no such graph")
        driver.close = AsyncMock()
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            resp = client.get("/admin/memory/abc/overview")
        assert resp.status_code == 200
        body = resp.json()
        assert body["entities"] == 0
        assert body["communities"] == 0


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

    def test_limit_clamps_at_500(self) -> None:
        resp = client.get("/admin/memory/abc/entities?limit=9999")
        # 422 from FastAPI validation, NOT 500
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


class TestRebuildCommunities:
    def test_forwards_to_scheduler_with_default_force_false(self) -> None:
        result_dict = {
            "user_id": "abc",
            "started_at": "2026-05-19T10:00:00Z",
            "communities_built": {"nodes": 30, "edges": 107},
            "elapsed_seconds": 18.7,
            "error": None,
            "skipped": False,
            "skip_reason": None,
            "activity": {"new_episodes_since_last_rebuild": 16},
            "forced": False,
        }
        scheduler = MagicMock()
        scheduler.execute_community_rebuild_pass = AsyncMock(return_value=result_dict)
        with patch(f"{_MOCK_MODULE}.get_scheduler_client", return_value=scheduler):
            resp = client.post("/admin/memory/abc/communities/rebuild")
        assert resp.status_code == 200
        scheduler.execute_community_rebuild_pass.assert_awaited_once_with(
            user_id="abc", force=False
        )
        body = resp.json()
        assert body["communities_built"] == {"nodes": 30, "edges": 107}
        assert body["activity"]["new_episodes_since_last_rebuild"] == 16

    def test_force_query_param_propagated(self) -> None:
        scheduler = MagicMock()
        scheduler.execute_community_rebuild_pass = AsyncMock(
            return_value={"user_id": "abc", "forced": True}
        )
        with patch(f"{_MOCK_MODULE}.get_scheduler_client", return_value=scheduler):
            resp = client.post("/admin/memory/abc/communities/rebuild?force=true")
        assert resp.status_code == 200
        scheduler.execute_community_rebuild_pass.assert_awaited_once_with(
            user_id="abc", force=True
        )

    def test_scheduler_failure_returns_500(self) -> None:
        scheduler = MagicMock()
        scheduler.execute_community_rebuild_pass = AsyncMock(
            side_effect=RuntimeError("scheduler unreachable")
        )
        with patch(f"{_MOCK_MODULE}.get_scheduler_client", return_value=scheduler):
            resp = client.post("/admin/memory/abc/communities/rebuild")
        assert resp.status_code == 500
        assert "scheduler unreachable" in resp.json()["detail"]


class TestAdminGating:
    """Non-admin callers must get 403 on every route."""

    def test_non_admin_gets_403(self, mock_jwt_user) -> None:
        # Swap to the non-admin user fixture
        app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
        try:
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
        finally:
            app.dependency_overrides[get_jwt_payload] = (
                pytest.importorskip("conftest", reason="reset to admin").mock_jwt_admin
            )
