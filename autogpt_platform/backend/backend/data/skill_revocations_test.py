from uuid import uuid4

import pytest
from prisma.models import SkillVersion

from backend.data import skill_versions as versions
from backend.data.skill_learning import LearningAccessError
from backend.data.skill_learning_test import _cleanup, _create_user, _source
from backend.util.json import SafeJson


async def _version(user, head, **kwargs):
    return await versions.create_version(
        user,
        head=head,
        content="Independent procedure",
        description="Test",
        triggers=[],
        origin="edited",
        **kwargs,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_revocation_covers_complete_lineage_and_is_repeatable_and_owned(server):
    user, foreign = str(uuid4()), str(uuid4())
    await _create_user(user)
    await _create_user(foreign)
    try:
        source = await _source(user)
        head = await versions.ensure_head(user, None, "revocation-check")
        independent = await _version(user, head)
        root = await _version(user, head, sources=[{"source_id": source.id}])
        bridge = await _version(user, head, base_version_id=root.id, state="archived")
        child = await _version(user, head, base_version_id=bridge.id)
        restored = await _version(
            user,
            head,
            base_version_id=child.id,
            restored_from_version_id=independent.id,
        )
        copied = await _version(user, head, base_version_id=restored.id)
        foreign_head = await versions.ensure_head(foreign, None, "foreign")
        foreign_version = await _version(
            foreign, foreign_head, sources=[{"source_id": source.id}]
        )
        noisy = await versions.ensure_head(user, None, "frequently-edited")
        await SkillVersion.prisma().create_many(
            data=[
                {
                    "id": str(uuid4()),
                    "userId": user,
                    "ownerKey": "personal",
                    "skillName": noisy.skill_name,
                    "headId": noisy.id,
                    "version": n,
                    "content": "Unrelated",
                    "contentHash": "noise",
                    "description": "Noise",
                    "triggers": [],
                    "origin": "edited",
                    "sources": SafeJson([]),
                    "evidence": SafeJson([]),
                    "limits": [],
                    "state": "ready",
                }
                for n in range(1, 602)
            ]
        )
        with pytest.raises(LearningAccessError):
            await versions.invalidate_source_versions(foreign, source.id, "withdrawn")
        expected = {root.id, bridge.id, child.id}
        for _ in range(2):
            affected = await versions.invalidate_source_versions(
                user, source.id, "withdrawn"
            )
            assert {v.id for v in affected} == expected
            assert all(v.head_id == head.id for v in affected)
        for version in (root, bridge, child):
            assert (await versions.get_version(user, version.id)).state == "invalidated"
        for version in (independent, restored, copied):
            assert (await versions.get_version(user, version.id)).state == "ready"
        assert (
            await versions.get_version(foreign, foreign_version.id)
        ).state == "ready"
        assert await versions.latest_independent_version_id(user, head.id) == copied.id
        assert await versions.latest_independent_version_id(foreign, head.id) is None
    finally:
        await _cleanup(user, foreign)
