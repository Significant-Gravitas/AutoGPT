from unittest.mock import AsyncMock

import pytest

from backend.api.features.skill_learning.routes_test import (
    EXPERT,
    _publish,
    client,
    run,
)
from backend.api.features.skill_learning.routes_test import (
    setup_app_auth as setup_app_auth,
)
from backend.api.features.skill_learning.routes_test import (
    store as learning_store_fixture,
)
from backend.data.skill_version_files import SkillVersionFile

store = learning_store_fixture


@pytest.mark.usefixtures("setup_app_auth")
def test_detail_fetches_files_only_for_current_selected_proposal_and_bases(
    store, test_user_id, monkeypatch
):
    saved = []
    for index in range(8):
        _, version = run(_publish(store, test_user_id, body=f"Revision {index}"))
        store.versions[version.id] = store.versions[version.id].model_copy(
            update={
                "files": [
                    SkillVersionFile.from_content(
                        "scripts/check.py", str(index).encode()
                    )
                ]
            }
        )
        saved.append(version)
    proposal = store.versions[saved[4].id]
    store.versions[proposal.id] = proposal.model_copy(
        update={"state": "needs_decision"}
    )
    fetch = AsyncMock(wraps=store.get_version)
    monkeypatch.setattr(store, "get_version", fetch)
    response = client.get(
        "/skill-learning/skills/csv-import-checks",
        params={
            "expert_id": EXPERT,
            "version_id": saved[1].id,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    by_id = {v["id"]: v for v in payload["versions"]}
    expected = {saved[i].id for i in (0, 1, 3, 4, 6, 7)}
    assert {c.args[1] for c in fetch.await_args_list} == expected
    assert fetch.await_count == len(expected)
    assert {v["id"] for v in payload["versions"] if v["files"]} == expected
    assert payload["current_version"]["files"][0]["content"] == "7"
    assert payload["open_decision"]["files"][0]["content"] == "4"
    assert by_id[saved[1].id]["files"][0]["content"] == "1"


@pytest.mark.usefixtures("setup_app_auth")
def test_selected_proposal_outside_history_still_has_its_decision_and_files(
    store, test_user_id, monkeypatch
):
    _, proposal = run(_publish(store, test_user_id))
    _, current = run(_publish(store, test_user_id))
    store.versions[proposal.id] = store.versions[proposal.id].model_copy(
        update={
            "state": "needs_decision",
            "files": [
                SkillVersionFile.from_content("scripts/check.py", b"print('proposed')")
            ],
        }
    )
    monkeypatch.setattr(store, "list_versions", AsyncMock(return_value=[current]))
    response = client.get(
        "/skill-learning/skills/csv-import-checks",
        params={
            "expert_id": EXPERT,
            "version_id": proposal.id,
        },
    )
    assert response.status_code == 200
    decision = response.json()["open_decision"]
    assert decision and decision["id"] == proposal.id
    assert decision["files"][0]["content"] == "print('proposed')"
