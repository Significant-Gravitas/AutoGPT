"""Tests for the user-facing memory routes (Settings → Memory)."""

from unittest.mock import AsyncMock, patch

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from redis.exceptions import ResponseError

from backend.api.features.experts.models import PROTECTED_SOUL_RULES, Expert
from backend.copilot.graphiti.client import derive_memory_group_id

from .routes import router as memory_router

app = fastapi.FastAPI()
app.include_router(memory_router)

client = fastapi.testclient.TestClient(app)

_MOCK_MODULE = "backend.api.features.memory.routes"

_EXPERT_ID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def memory_enabled():
    with patch(
        f"{_MOCK_MODULE}.is_enabled_for_user", new=AsyncMock(return_value=True)
    ) as mock:
        yield mock


def _driver_returning(*query_results) -> AsyncMock:
    driver = AsyncMock()
    driver.execute_query.side_effect = [(r, None, None) for r in query_results]
    driver.close = AsyncMock()
    return driver


def _expert(expert_id: str = _EXPERT_ID) -> Expert:
    return Expert(
        id=expert_id,
        name="Maria",
        avatar_url=None,
        role="Growth Marketer",
        tagline=None,
        bio=None,
        skills=[],
        identity="Growth expert.",
        voice_preferences="",
        boundaries="",
        protected_soul_rules=list(PROTECTED_SOUL_RULES),
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=[],
    )


class TestOverview:
    def test_returns_scope_counts(self) -> None:
        driver = _driver_returning([{"c": 12}], [{"c": 34}], [{"c": 5}])
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            resp = client.get("/memory/overview")
        assert resp.status_code == 200
        body = resp.json()
        assert body["expert_id"] is None
        assert body["facts"] == 12
        assert body["entities"] == 34
        assert body["episodes"] == 5
        driver.close.assert_awaited_once()

    def test_disabled_memory_is_403(self, memory_enabled) -> None:
        memory_enabled.return_value = False
        resp = client.get("/memory/overview")
        assert resp.status_code == 403

    def test_fact_count_only_counts_live_edges(self) -> None:
        driver = _driver_returning([{"c": 0}], [{"c": 0}], [{"c": 0}])
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            client.get("/memory/overview")
        fact_query = driver.execute_query.await_args_list[0].args[0]
        assert "expired_at IS NULL" in fact_query


class TestListFacts:
    def test_maps_rows_to_facts(self) -> None:
        driver = _driver_returning(
            [
                {
                    "uuid": "edge-1",
                    "fact": "Prefers Monday summaries",
                    "name": "prefers",
                    "source": "User",
                    "target": "Monday summaries",
                    "created_at": "2026-08-16T00:00:00Z",
                }
            ]
        )
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            resp = client.get("/memory/facts?limit=5")
        assert resp.status_code == 200
        body = resp.json()
        assert body["expert_id"] is None
        assert body["items"][0]["uuid"] == "edge-1"
        assert body["items"][0]["fact"] == "Prefers Monday summaries"
        query = driver.execute_query.await_args_list[0].args[0]
        assert "expired_at IS NULL" in query
        assert "ORDER BY e.created_at DESC" in query

    def test_missing_graph_is_empty_list(self) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = ResponseError("no such graph")
        driver.close = AsyncMock()
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            resp = client.get("/memory/facts")
        assert resp.status_code == 200
        assert resp.json()["items"] == []

    def test_expert_scope_uses_expert_group(self, test_user_id) -> None:
        driver = _driver_returning([])
        opened: list[str] = []

        def open_driver(group_id: str):
            opened.append(group_id)
            return driver

        with (
            patch(f"{_MOCK_MODULE}._open_driver", side_effect=open_driver),
            patch(
                f"{_MOCK_MODULE}.experts_db.get_expert",
                new=AsyncMock(return_value=_expert()),
            ) as get_expert,
        ):
            resp = client.get(f"/memory/experts/{_EXPERT_ID}/facts")
        assert resp.status_code == 200
        assert resp.json()["expert_id"] == _EXPERT_ID
        assert opened == [derive_memory_group_id(test_user_id, _EXPERT_ID)]
        get_expert.assert_awaited_once_with(
            test_user_id, _EXPERT_ID, include_workflows=False
        )

    def test_unknown_expert_is_404(self) -> None:
        with patch(
            f"{_MOCK_MODULE}.experts_db.get_expert",
            new=AsyncMock(return_value=None),
        ):
            resp = client.get(f"/memory/experts/{_EXPERT_ID}/facts")
        assert resp.status_code == 404


class TestForgetFact:
    def test_retracts_matching_edge(self) -> None:
        driver = _driver_returning([{"uuid": "edge-1"}])
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            resp = client.delete("/memory/facts/edge-1")
        assert resp.status_code == 200
        assert resp.json() == {"uuid": "edge-1", "forgotten": True}
        query = driver.execute_query.await_args_list[0].args[0]
        assert "SET e.expired_at = $now" in query
        assert "group_id: $g" in query
        assert "DELETE" not in query

    def test_no_match_is_404(self) -> None:
        driver = _driver_returning([])
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            resp = client.delete("/memory/facts/edge-unknown")
        assert resp.status_code == 404


class TestEraseScope:
    def test_erases_all_nodes(self) -> None:
        driver = _driver_returning([{"c": 214}], [])
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            resp = client.delete("/memory")
        assert resp.status_code == 200
        body = resp.json()
        assert body["erased"] is True
        assert body["deleted_nodes"] == 214
        wipe_query = driver.execute_query.await_args_list[1].args[0]
        assert "DETACH DELETE" in wipe_query

    def test_empty_scope_never_touches_graph(self) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = ResponseError("no such graph")
        driver.close = AsyncMock()
        with patch(f"{_MOCK_MODULE}._open_driver", return_value=driver):
            resp = client.delete("/memory")
        assert resp.status_code == 200
        assert resp.json()["deleted_nodes"] == 0
        assert driver.execute_query.await_count == 1

    def test_expert_erase_resolves_ownership(self, test_user_id) -> None:
        driver = _driver_returning([{"c": 3}], [])
        with (
            patch(f"{_MOCK_MODULE}._open_driver", return_value=driver),
            patch(
                f"{_MOCK_MODULE}.experts_db.get_expert",
                new=AsyncMock(return_value=_expert()),
            ) as get_expert,
        ):
            resp = client.delete(f"/memory/experts/{_EXPERT_ID}")
        assert resp.status_code == 200
        assert resp.json()["expert_id"] == _EXPERT_ID
        get_expert.assert_awaited_once_with(
            test_user_id, _EXPERT_ID, include_workflows=False
        )
