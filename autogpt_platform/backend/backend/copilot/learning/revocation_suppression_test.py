from unittest.mock import AsyncMock

import pytest

from backend.data.skill_publication import VersionDraft

from . import owner_actions, publish, publish_test, revocation
from .contract import register_source_adapter
from .publish_test import BODY_V1, BODY_V2, EXPERT, KIND, NAME, USER, _Adapter, _content

workspace = publish_test.workspace


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_restore", [False, True])
async def test_source_revocation_does_not_reject_independently_learned_behavior(
    fake_store, workspace, monkeypatch, owner_restore
):
    register_source_adapter(_Adapter(fake_store))
    sources = [
        await fake_store.upsert_source_revision(
            USER,
            expert_id=EXPERT,
            source_kind=KIND,
            source_id=ref,
            revision="1",
            evidence_refs=[],
            outcome_signals=[],
        )
        for ref in ("withdrawn", "independent")
    ]
    snapshots = [
        publish.SourceSnapshot(
            source_id=s.id,
            source_kind=s.source_kind,
            source_ref=s.source_id,
            revision=s.revision,
            epoch=s.epoch,
        )
        for s in sources
    ]
    versions = []
    for body, dependencies in (
        (BODY_V1, []),
        (BODY_V2, [snapshots[0].as_dependency()]),
    ):
        head = await fake_store.ensure_head(USER, EXPERT, NAME)
        committed = await fake_store.commit_version_safe(
            USER,
            head=head,
            expected_current_version=head.current_version,
            draft=VersionDraft(
                content=_content(body),
                description="Import a CSV",
                triggers=["csv"],
                origin="saved_overnight",
                base_version_id=head.current_version_id,
                sources=dependencies,
            ),
        )
        await fake_store.complete_publication(
            USER, version_id=committed.version.id, review_id=None
        )
        versions.append(committed.version)

    if owner_restore:
        restored = await owner_actions.restore_version(
            user_id=USER,
            expert_id=EXPERT,
            skill_name=NAME,
            version_id=versions[0].id,
            actor_user_id=USER,
        )
        assert restored.status == "applied"
    else:
        excluded = await fake_store.set_source_eligibility(
            USER, sources[0].id, "excluded", excluded_by_user_id=USER
        )
        affected = await revocation.invalidate_versions_for_source(
            USER, excluded, "source excluded"
        )
        assert affected == [versions[1].id]
        assert (
            await fake_store.get_version(USER, versions[1].id)
        ).state == "invalidated"

    head = await fake_store.get_head(USER, EXPERT, NAME)
    current = await fake_store.get_version(USER, head.current_version_id)
    assert current.restored_from_version_id == versions[0].id
    assert current.content == _content(BODY_V1)
    assert not head.auto_improve
    monkeypatch.setattr(
        publish, "read_user_skill_markdown", AsyncMock(return_value=current.content)
    )
    request = publish.PublishRequest(
        user_id=USER,
        expert_id=EXPERT,
        skill_name=NAME,
        description="Import a CSV",
        triggers=["csv"],
        body=BODY_V2,
        summary="Independently verified row validation",
        origin="saved_overnight",
        sources=[snapshots[1]],
    )
    outcome = await publish.publish_learned_version(request)

    assert outcome.status == ("suppressed" if owner_restore else "needs_decision")
    suppressions = await fake_store.list_suppressions(USER, EXPERT, NAME)
    assert len(suppressions) == int(owner_restore)
    if not owner_restore:
        assert outcome.version.sources == [snapshots[1].as_dependency()]
        withdrawn = await publish.publish_learned_version(
            request.model_copy(update={"sources": [snapshots[0]]})
        )
        assert withdrawn.status == "stale_eligibility"
