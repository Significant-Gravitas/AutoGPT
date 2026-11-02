import pytest

from backend.api.features.store.exceptions import (
    StorageUploadError,
    VirusDetectedError,
    VirusScanError,
)
from backend.data.skill_capacity import SkillLimitError

from . import nightly_test
from .nightly_test import EXPERT, USER, _run, _source
from .publish_test import NAME, _seed
from .untracked_edit_recovery_test import _review_edit

adapter = nightly_test.adapter
boundaries = nightly_test.boundaries


@pytest.mark.asyncio
@pytest.mark.parametrize("during_reconcile", [False, True])
async def test_virus_rejection_abandons_the_snapshot_without_repeat_review(
    fake_store, adapter, boundaries, during_reconcile
):
    source = await _source(fake_store)
    if during_reconcile:
        boundaries["write"].side_effect = VirusScanError("scanner unavailable")
        assert (await _run()).dispositions == {"write_failed": 1}
    boundaries["write"].side_effect = VirusDetectedError("test-threat")
    rejected = await _run()
    assert rejected.dispositions == {"blocked_content": 1}
    versions = await fake_store.list_versions(USER, EXPERT, NAME)
    assert len(versions) == 1 and versions[0].state == "stale"
    head = await fake_store.get_head(USER, EXPERT, NAME)
    assert head.current_version_id == versions[0].id
    ledger = await fake_store.get_review_for_revision(
        USER, source.id, source.revision, 1
    )
    assert ledger.disposition == "blocked_content"
    assert (
        await fake_store.get_source(USER, source.id)
    ).processed_revision == source.revision
    assert await fake_store.list_pending_publications(USER) == []
    attempts = boundaries["write"].await_count
    assert (await _run()).skip_reason == "no_eligible_work"
    assert boundaries["write"].await_count == attempts
    boundaries["review"].assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error, disposition",
    [
        (VirusScanError("scanner unavailable"), "write_failed"),
        (StorageUploadError("upload unavailable"), "write_failed"),
        (SkillLimitError("skill capacity reached"), "paused"),
        (ValueError("storage quota exceeded"), "write_failed"),
    ],
    ids=["scanner", "storage", "capacity", "quota"],
)
async def test_recoverable_publication_failure_replays_without_model_calls(
    fake_store, adapter, boundaries, error, disposition
):
    source = await _source(fake_store)
    boundaries["write"].side_effect = error
    assert (await _run()).dispositions == {disposition: 1}
    version = (await fake_store.list_pending_publications(USER))[0]
    assert (await fake_store.get_source(USER, source.id)).has_unprocessed_revision
    boundaries["write"].side_effect = None
    recovered = await _run()
    assert recovered.reconciled_writes == 1 and recovered.model_calls == 0
    assert (await fake_store.get_version(USER, version.id)).state == "ready"
    boundaries["review"].assert_awaited_once()


@pytest.mark.asyncio
async def test_untracked_edit_virus_rejection_is_terminal(
    fake_store, adapter, boundaries, monkeypatch
):
    await _seed(fake_store)
    source = await _source(fake_store)
    _review_edit(monkeypatch, boundaries)
    boundaries["write"].side_effect = VirusDetectedError("test-threat")
    assert (await _run()).dispositions == {"blocked_content": 1}
    assert (
        await fake_store.get_source(USER, source.id)
    ).processed_revision == source.revision
    assert len(await fake_store.list_versions(USER, EXPERT, NAME)) == 2
    assert (await _run()).skip_reason == "no_eligible_work"
    boundaries["review"].assert_awaited_once()
