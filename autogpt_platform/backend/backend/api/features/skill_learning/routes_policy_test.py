from unittest.mock import AsyncMock

import pytest

from . import routes, routes_test

store = routes_test.store
setup_app_auth = routes_test.setup_app_auth


@pytest.mark.parametrize("scope", [None, routes_test.EXPERT])
@pytest.mark.parametrize("collision", ["missing", "foreign-account", "other-scope"])
def test_policy_requires_a_head_owned_by_the_requested_scope(
    store, test_user_id, scope, collision
):
    if collision == "foreign-account":
        routes_test.run(store.ensure_head("another-user", scope, "checks"))
    elif collision == "other-scope":
        other_scope = None if scope else routes_test.EXPERT
        routes_test.run(store.ensure_head(test_user_id, other_scope, "checks"))
    before = dict(store.heads)
    response = routes_test.client.post(
        "/skill-learning/skills/checks/policy",
        params={"expert_id": scope} if scope else {},
        json={"auto_improve": False},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Skill not found"
    assert store.heads == before


@pytest.mark.parametrize("scope", [None, routes_test.EXPERT])
def test_policy_can_update_an_owned_head_before_its_first_publication(
    store, test_user_id, scope, monkeypatch
):
    invalidate = AsyncMock()
    monkeypatch.setattr(routes, "invalidate_skills_index_cache", invalidate)
    head = routes_test.run(store.ensure_head(test_user_id, scope, "checks"))
    assert head.current_version_id is None
    response = routes_test.client.post(
        "/skill-learning/skills/CHECKS/policy",
        params={"expert_id": scope} if scope else {},
        json={"auto_improve": False, "learning_paused": True, "use_paused": True},
    )
    assert response.status_code == 200
    assert response.json() == {
        "auto_improve": False,
        "learning_paused": True,
        "use_paused": True,
    }
    assert len(store.heads) == 1
    updated = routes_test.run(store.get_head(test_user_id, head.owner_key, "checks"))
    assert updated.id == head.id and updated.current_version_id is None
    invalidate.assert_awaited_once_with(test_user_id, scope)


def test_policy_refuses_an_expert_outside_the_callers_scope(store, monkeypatch):
    monkeypatch.setattr(
        routes.experts_db, "owns_private_active_expert", AsyncMock(return_value=False)
    )
    response = routes_test.client.post(
        "/skill-learning/skills/checks/policy",
        params={"expert_id": "foreign-expert"},
        json={"use_paused": True},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Expert not found"
    assert not store.heads
