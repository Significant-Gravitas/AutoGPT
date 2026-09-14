"""Route tests for the skill-learning API (in-memory store, mocked registry)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.copilot.learning import owner_actions, publish, revocation
from backend.copilot.learning._fake_store import FakeLearningStore
from backend.copilot.learning.chat_source import CHAT_SOURCE_KIND
from backend.copilot.model import ChatSessionInfo
from backend.copilot.tools.skills import ParsedSkill, render_skill_markdown
from backend.data.skill_publication import VersionDraft

from . import routes, views
from .routes import router


def run(coro):
    return asyncio.run(coro)


app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

EXPERT = "expert-1"
SESSION = "session-1"
BODY = "## Steps\n1. Open the file as utf-8\n\n## Verification\nrows match\n"


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def store(monkeypatch, test_user_id):
    fake = FakeLearningStore()
    monkeypatch.setattr(routes, "learning_data", fake)
    monkeypatch.setattr(routes, "reviews_data", fake)
    monkeypatch.setattr(routes, "versions_data", fake)
    monkeypatch.setattr(routes, "use_data", fake)
    monkeypatch.setattr(revocation, "invalidate_skills_index_cache", AsyncMock())
    monkeypatch.setattr(views, "learning_data", fake)
    for module in (publish, owner_actions, revocation):
        for seam in ("skill_versions_db", "skill_publication_db", "skill_use_db"):
            monkeypatch.setattr(module, seam, lambda: fake, raising=False)
    monkeypatch.setattr(routes, "invalidate_skills_index_cache", AsyncMock())
    monkeypatch.setattr(publish, "invalidate_skills_index_cache", AsyncMock())
    monkeypatch.setattr(publish, "store_user_skill", AsyncMock())
    monkeypatch.setattr(owner_actions, "store_user_skill", AsyncMock())
    monkeypatch.setattr(publish, "read_skill_bundle_files", AsyncMock(return_value={}))
    monkeypatch.setattr(
        owner_actions, "read_user_skill_with_body", AsyncMock(return_value=None)
    )
    experts = AsyncMock()
    experts.owns_private_active_expert = AsyncMock(return_value=True)
    experts.get_expert = AsyncMock(return_value=None)
    monkeypatch.setattr(routes, "experts_db", experts)
    monkeypatch.setattr(routes, "is_feature_enabled", AsyncMock(return_value=True))
    now = datetime.now(timezone.utc)
    session = ChatSessionInfo(
        session_id=SESSION,
        user_id=test_user_id,
        usage=[],
        started_at=now,
        updated_at=now,
        expert_id=EXPERT,
        title="Fix the CSV import",
    )
    monkeypatch.setattr(
        views.chat_db, "get_chat_session_metadata", AsyncMock(return_value=session)
    )
    # The real chat adapter revalidates sources during restore/decide; route
    # it at the in-memory store and a stubbed chat/expert lookup.
    from backend.copilot.learning import chat_source

    chat = AsyncMock()
    chat.get_chat_session_metadata = AsyncMock(return_value=session)
    expert = AsyncMock()
    expert.learning_paused_at = None
    expert_lookup = AsyncMock()
    expert_lookup.get_expert = AsyncMock(return_value=expert)
    monkeypatch.setattr(chat_source, "skill_learning_db", lambda: fake)
    monkeypatch.setattr(chat_source, "chat_db", lambda: chat)
    monkeypatch.setattr(chat_source, "experts_db", lambda: expert_lookup)
    return fake


async def _publish(store: FakeLearningStore, user_id: str, *, body: str = BODY):
    source = await store.upsert_source_revision(
        user_id,
        expert_id=EXPERT,
        source_kind=CHAT_SOURCE_KIND,
        source_id=SESSION,
        revision="4",
        evidence_refs=[],
        outcome_signals=[{"kind": "tool_result", "ref": "msg:3", "label": "ok"}],
    )
    head = await store.ensure_head(user_id, EXPERT, "csv-import-checks")
    result = await store.commit_version_safe(
        user_id,
        head=head,
        draft=VersionDraft(
            content=render_skill_markdown(
                ParsedSkill(
                    name="csv-import-checks", description="Import CSV", body=body
                )
            ),
            description="Import CSV",
            origin="saved_overnight",
            base_version_id=head.current_version_id,
            summary="Added an encoding check after the previous import failed.",
            sources=[
                {
                    "source_id": source.id,
                    "source_kind": CHAT_SOURCE_KIND,
                    "source_ref": SESSION,
                    "revision": source.revision,
                    "epoch": 0,
                }
            ],
            evidence=[
                {"kind": "outcome", "ref": "", "label": "Worked once in the source"}
            ],
        ),
        expected_current_version=head.current_version,
    )
    await store.complete_publication(
        user_id, version_id=result.version.id, review_id=None
    )
    return source, result.version


def test_detail_shows_version_evidence_source_and_reuse(store, test_user_id):
    source, version = run(_publish(store, test_user_id))
    run(
        store.record_use_event(
            test_user_id,
            expert_id=EXPERT,
            skill_name="csv-import-checks",
            kind="loaded",
            version_id=version.id,
        )
    )
    resp = client.get(
        "/skill-learning/skills/csv-import-checks", params={"expert_id": EXPERT}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["state_label"] == "Ready to use"
    assert body["current_version"]["version"] == 1
    assert body["current_version"]["origin_label"] == "Saved overnight"
    assert (
        body["current_version"]["evidence"][0]["label"] == "Worked once in the source"
    )
    link = body["current_version"]["sources"][0]
    assert link["accessible"] and link["title"] == "Fix the CSV import"
    assert link["url"] == f"/copilot?sessionId={SESSION}&expertId={EXPERT}"
    assert body["use"]["reuse_label"] == "Loaded 1 time · Outcome unknown"
    assert body["policy"] == {
        "auto_improve": True,
        "learning_paused": False,
        "use_paused": False,
    }


def test_inaccessible_source_leaks_no_title_or_link(store, test_user_id, monkeypatch):
    run(_publish(store, test_user_id))
    monkeypatch.setattr(
        views.chat_db, "get_chat_session_metadata", AsyncMock(return_value=None)
    )
    resp = client.get(
        "/skill-learning/skills/csv-import-checks", params={"expert_id": EXPERT}
    )
    link = resp.json()["current_version"]["sources"][0]
    assert link == {
        "source_id": link["source_id"],
        "source_kind": CHAT_SOURCE_KIND,
        "revision": "000000000004",
        "accessible": False,
        "source_ref": None,
        "title": None,
        "url": None,
        "excluded": False,
    }
    assert "Fix the CSV import" not in resp.text
    assert resp.json()["state"] == "ready"  # hidden evidence never pauses the skill


def test_foreign_expert_scope_is_not_found(store):
    store_experts = routes.experts_db
    store_experts.owns_private_active_expert.return_value = False
    resp = client.get(
        "/skill-learning/skills/csv-import-checks", params={"expert_id": "other"}
    )
    assert resp.status_code == 404


def test_restore_creates_a_new_version_and_history_shows_it(store, test_user_id):
    _, v1 = run(_publish(store, test_user_id))
    _, v2 = run(
        _publish(store, test_user_id, body=BODY + "2. Validate the row count\n")
    )
    resp = client.post(
        "/skill-learning/skills/csv-import-checks/restore",
        params={"expert_id": EXPERT},
        json={"version_id": v1.id},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["status"] == "applied" and payload["version"]["origin"] == "restored"
    assert payload["version"]["restored_from_version_id"] == v1.id
    history = client.get("/skill-learning/history", params={"expert_id": EXPERT}).json()
    origins = [item["origin"] for item in history["items"] if item["kind"] == "version"]
    assert origins[0] == "restored"
    detail = client.get(
        "/skill-learning/skills/csv-import-checks", params={"expert_id": EXPERT}
    ).json()
    assert detail["policy"]["auto_improve"] is False


def test_edit_with_a_stale_base_is_a_conflict_that_keeps_the_newer_version(
    store, test_user_id, monkeypatch
):
    """The editor opened v1; v2 landed while they typed. Their submission
    names v1 as its base and must not replace v2 (200 + conflict, so the
    client keeps the draft)."""
    _, v1 = run(_publish(store, test_user_id))
    _, v2 = run(
        _publish(store, test_user_id, body=BODY + "2. Validate the row count\n")
    )
    from backend.copilot.tools.skills import SkillVersionConflictError

    async def guarded_write(user_id, **kwargs):
        head = await store.get_head(user_id, EXPERT, kwargs["name"])
        expected = kwargs["expected_head"].version_id
        if head.current_version_id != expected:
            raise SkillVersionConflictError(expected, head.current_version_id)
        raise AssertionError("a stale base must never reach the write")

    monkeypatch.setattr(
        owner_actions, "store_user_skill", AsyncMock(side_effect=guarded_write)
    )
    resp = client.post(
        "/skill-learning/skills/csv-import-checks/edit",
        params={"expert_id": EXPERT},
        json={
            "description": "Import CSV",
            "body": "## Steps\n1. typed over v1\n",
            "triggers": [],
            "expected_version_id": v1.id,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["status"] == "conflict" and payload["version"] is None
    assert "changed since you started editing" in payload["reason"]
    head = run(store.get_head(test_user_id, EXPERT, "csv-import-checks"))
    assert head.current_version_id == v2.id


def test_edit_blocked_by_content_check_returns_only_safe_diagnostics(
    store, test_user_id, monkeypatch
):
    run(_publish(store, test_user_id))
    from backend.copilot.learning.content_checks import check_skill_content
    from backend.copilot.tools.skills import SkillContentBlockedError

    secret = "ghp_" + "z" * 40

    async def blocked_write(*_args, **kwargs):
        failure = check_skill_content(kwargs["body"])
        assert failure is not None
        raise SkillContentBlockedError(failure)

    monkeypatch.setattr(
        owner_actions, "store_user_skill", AsyncMock(side_effect=blocked_write)
    )
    resp = client.post(
        "/skill-learning/skills/csv-import-checks/edit",
        params={"expert_id": EXPERT},
        json={
            "description": "Import CSV",
            "body": f"## Steps\n1. token: {secret}\n",
            "triggers": [],
            "expected_version_id": run(
                store.get_head(test_user_id, EXPERT, "csv-import-checks")
            ).current_version_id,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "blocked_content"
    assert payload["pattern_class"] == "github_token"
    assert payload["version"] is None
    assert "github_token" in resp.text and secret not in resp.text


def test_policy_endpoints_keep_pause_learning_and_stop_using_separate(
    store, test_user_id
):
    run(_publish(store, test_user_id))
    resp = client.post(
        "/skill-learning/skills/csv-import-checks/policy",
        params={"expert_id": EXPERT},
        json={"learning_paused": True},
    )
    assert resp.json() == {
        "auto_improve": True,
        "learning_paused": True,
        "use_paused": False,
    }
    resp = client.post(
        "/skill-learning/skills/csv-import-checks/policy",
        params={"expert_id": EXPERT},
        json={"use_paused": True, "learning_paused": False},
    )
    assert resp.json() == {
        "auto_improve": True,
        "learning_paused": False,
        "use_paused": True,
    }
    detail = client.get(
        "/skill-learning/skills/csv-import-checks", params={"expert_id": EXPERT}
    ).json()
    assert detail["state_label"] == "Paused (not in use)"


def test_exclude_source_invalidates_dependents_and_pauses_without_fallback(
    store, test_user_id
):
    source, version = run(_publish(store, test_user_id))
    resp = client.post(f"/skill-learning/sources/{CHAT_SOURCE_KIND}/{SESSION}/exclude")
    assert resp.status_code == 200, resp.text
    assert resp.json()["invalidated_version_ids"] == [version.id]
    assert run(store.get_source(test_user_id, source.id)).eligibility == "excluded"
    detail = client.get(
        "/skill-learning/skills/csv-import-checks", params={"expert_id": EXPERT}
    ).json()
    assert detail["policy"]["use_paused"] is True
    assert detail["versions"][0]["state_label"] == "Archived (source unavailable)"
    assert detail["versions"][0]["sources"][0]["excluded"] is True


def test_decisions_list_and_outcome_report(store, test_user_id):
    _, version = run(_publish(store, test_user_id))
    head = run(store.get_head(test_user_id, EXPERT, "csv-import-checks"))
    proposal = run(
        store.create_version(
            test_user_id,
            head=head,
            content=version.content + "\n3. Log it\n",
            description="Import CSV",
            triggers=[],
            origin="saved_overnight",
            state="needs_decision",
            state_reason="automatic improvements are off for this skill",
            base_version_id=head.current_version_id,
        )
    )
    decisions = client.get("/skill-learning/decisions").json()
    assert [d["id"] for d in decisions["items"]] == [proposal.id]
    assert decisions["items"][0]["state_label"] == "Needs your decision"
    resp = client.post(
        f"/skill-learning/skills/csv-import-checks/decisions/{proposal.id}",
        params={"expert_id": EXPERT},
        json={"action": "keep_current"},
    )
    assert resp.status_code == 200 and resp.json()["status"] == "applied"
    assert client.get("/skill-learning/decisions").json()["items"] == []

    resp = client.post(
        "/skill-learning/skills/csv-import-checks/outcome",
        params={"expert_id": EXPERT},
        json={
            "version_id": version.id,
            "outcome": "succeeded",
            "detail": "row count matched",
        },
    )
    assert resp.status_code == 200
    detail = client.get(
        "/skill-learning/skills/csv-import-checks", params={"expert_id": EXPERT}
    ).json()
    assert detail["use"]["reuse_label"] == "Reported as working by you (1 report)"
    assert detail["use"]["checks_passed"] == 0


def test_status_reports_backlog_and_flag(store):
    resp = client.get("/skill-learning/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True and body["pending_sources"] == 0
    assert CHAT_SOURCE_KIND in body["source_kinds"]


def test_run_now_uses_the_same_pass(store, monkeypatch):
    from backend.copilot.learning.nightly import SkillLearningResult

    fake = AsyncMock(
        return_value=SkillLearningResult(
            user_id="u",
            run_id="r",
            trigger="admin",
            started_at=datetime.now(timezone.utc),
            skipped=True,
            skip_reason="no_eligible_work",
        )
    )
    monkeypatch.setattr(routes, "run_skill_learning_pass", fake)
    resp = client.post("/skill-learning/run")
    assert resp.status_code == 200 and resp.json()["skip_reason"] == "no_eligible_work"
    assert fake.await_args.kwargs["trigger"] == "admin"


def test_expert_learning_pause_endpoint(store, monkeypatch, test_user_id):
    from backend.api.features.experts.models import PROTECTED_SOUL_RULES, Expert

    expert = Expert(
        id=EXPERT,
        name="Alex",
        avatar_url=None,
        role="Ops",
        tagline=None,
        bio=None,
        skills=[],
        identity="i",
        voice_preferences="",
        boundaries="",
        protected_soul_rules=list(PROTECTED_SOUL_RULES),
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=[],
        learning_paused_at=datetime.now(timezone.utc),
    )
    routes.experts_db.set_expert_learning_paused = AsyncMock(return_value=expert)
    resp = client.post(
        f"/skill-learning/experts/{EXPERT}/policy", json={"learning_paused": True}
    )
    assert resp.status_code == 200 and resp.json()["learning_paused_at"] is not None
    routes.experts_db.set_expert_learning_paused = AsyncMock(return_value=None)
    assert (
        client.post(
            "/skill-learning/experts/nope/policy", json={"learning_paused": True}
        ).status_code
        == 404
    )


def test_detail_includes_requested_version_outside_recent_history(store, test_user_id):
    _, oldest = run(_publish(store, test_user_id))
    for i in range(51):
        _, current = run(_publish(store, test_user_id, body=BODY + f"Revision {i}"))
    resp = client.get(
        "/skill-learning/skills/csv-import-checks",
        params={"expert_id": EXPERT, "version_id": oldest.id},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["current_version"]["id"] == current.id
    assert oldest.id in {v["id"] for v in payload["versions"]}


@pytest.mark.parametrize("scope", [None, "other-expert"])
def test_detail_rejects_version_from_another_owned_scope(store, test_user_id, scope):
    _, version = run(_publish(store, test_user_id))
    run(store.ensure_head(test_user_id, scope, "csv-import-checks"))
    params = {"version_id": version.id}
    if scope:
        params["expert_id"] = scope
    resp = client.get("/skill-learning/skills/csv-import-checks", params=params)
    assert resp.status_code == 404
    assert "Import CSV" not in resp.text


def test_detail_rejects_missing_version_instead_of_showing_current(store, test_user_id):
    run(_publish(store, test_user_id))
    resp = client.get(
        "/skill-learning/skills/csv-import-checks",
        params={"expert_id": EXPERT, "version_id": "missing"},
    )
    assert resp.status_code == 404


def test_detail_includes_the_comparison_base_of_an_old_link(store, test_user_id):
    _, base = run(_publish(store, test_user_id))
    _, selected = run(_publish(store, test_user_id, body=BODY + "2. Verify types"))
    for i in range(51):
        run(_publish(store, test_user_id, body=BODY + f"Revision {i}"))
    payload = client.get(
        "/skill-learning/skills/csv-import-checks",
        params={"expert_id": EXPERT, "version_id": selected.id},
    ).json()
    by_id = {version["id"]: version for version in payload["versions"]}
    assert by_id[selected.id]["base_version_id"] == base.id
    assert by_id[base.id]["body"] == base.content


def test_detail_does_not_lose_current_version_behind_unpublished_history(
    store, test_user_id
):
    _, current = run(_publish(store, test_user_id))
    head = run(store.get_head(test_user_id, EXPERT, "csv-import-checks"))
    for i in range(51):
        run(
            store.create_version(
                test_user_id,
                head=head,
                content=f"Attempt {i}",
                description="Unpublished",
                triggers=[],
                origin="saved_overnight",
                state="archived",
                base_version_id=current.id,
            )
        )
    resp = client.get(
        "/skill-learning/skills/csv-import-checks", params={"expert_id": EXPERT}
    )
    assert resp.status_code == 200
    assert resp.json()["current_version"]["id"] == current.id
    assert resp.json()["state"] == "ready"
