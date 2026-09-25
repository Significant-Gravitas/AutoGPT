from unittest.mock import AsyncMock

import pytest

from backend.api.features.experts import hire_skill_snapshot
from backend.api.features.experts.errors import ExpertTemplateNotFoundError


@pytest.mark.asyncio
async def test_capture_holds_catalogue_read_lock_before_reading_assignments(
    monkeypatch,
):
    query = AsyncMock(
        side_effect=[
            [{"activeReleaseId": "release-a"}],
            [{"slug": "research", "version_id": "v1", "title": "Research"}],
        ]
    )
    monkeypatch.setattr(hire_skill_snapshot, "query_raw_with_schema", query)
    tx = AsyncMock()

    snapshot = await hire_skill_snapshot.capture_skill_snapshot(tx, "template")

    assert snapshot.release_id == "release-a"
    assert [(p.slug, p.version_id) for p in snapshot.packages] == [("research", "v1")]
    assert "FOR SHARE" in query.await_args_list[0].args[0]
    assert all(call.kwargs["client"] is tx for call in query.await_args_list)


@pytest.mark.asyncio
async def test_withdrawn_template_cannot_create_a_hire_even_with_skills_disabled(
    monkeypatch,
):
    query = AsyncMock(return_value=[{"activeReleaseId": "release-a"}])
    monkeypatch.setattr(hire_skill_snapshot, "query_raw_with_schema", query)
    tx = AsyncMock()
    tx.expert.find_first.return_value = None
    with pytest.raises(ExpertTemplateNotFoundError):
        await hire_skill_snapshot.capture_skill_snapshot(tx, "withdrawn", enabled=False)
    assert query.await_count == 1


def test_legacy_hire_does_not_resolve_current_template():
    with pytest.raises(hire_skill_snapshot.LegacySkillSnapshotError):
        hire_skill_snapshot.read_skill_snapshot(None)


def test_snapshot_rejects_duplicate_package_names():
    with pytest.raises(ValueError, match="duplicate"):
        hire_skill_snapshot.read_skill_snapshot(
            {
                "release_id": "release-a",
                "packages": [
                    {"slug": "research", "version_id": "v1", "title": "Research"},
                    {"slug": "research", "version_id": "v2", "title": "Research"},
                ],
            }
        )
