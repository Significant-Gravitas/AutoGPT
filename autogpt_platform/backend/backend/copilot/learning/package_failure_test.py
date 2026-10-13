from unittest.mock import AsyncMock

import pytest

from backend.copilot.tools.skills import ParsedSkill, SkillPackageError

from . import nightly, nightly_test, publish
from .nightly_test import GOOD_BODY, USER, _completion, _proposal, _run, _source

adapter = nightly_test.adapter
boundaries = nightly_test.boundaries


def _existing_package(monkeypatch, boundaries):
    skill = ParsedSkill(
        name="csv-import-checks", description="CSV validation", body=GOOD_BODY
    )
    monkeypatch.setattr(nightly, "_existing_skills", AsyncMock(return_value=[skill]))
    monkeypatch.setattr(
        publish,
        "read_user_skill_markdown",
        AsyncMock(return_value=publish.render_skill_markdown(skill)),
    )
    boundaries["review"].return_value = _completion(
        _proposal(decision="update", files=None)
    )


@pytest.mark.asyncio
async def test_oversized_package_settles_body_only_update_without_repeated_review(
    fake_store, adapter, boundaries, monkeypatch
):
    _existing_package(monkeypatch, boundaries)
    source = await _source(fake_store)
    files = AsyncMock(
        side_effect=SkillPackageError(
            "stored package has more than 100 files and cannot be served whole",
            over_limit=True,
        )
    )
    monkeypatch.setattr(publish, "read_user_skill_files", files)

    first = await _run()

    assert first.dispositions == {"skipped": 1}
    ledger = await fake_store.get_review_for_revision(USER, source.id, "4", 1)
    assert ledger.disposition == "skipped"
    assert ledger.cost_microdollars == 1000
    assert not (await fake_store.get_source(USER, source.id)).has_unprocessed_revision
    boundaries["write"].assert_not_awaited()

    second = await _run()

    assert second.skip_reason == "no_eligible_work"
    assert second.model_calls == 0
    boundaries["review"].assert_awaited_once()
    files.assert_awaited_once()


@pytest.mark.asyncio
async def test_transient_package_read_failure_remains_retryable(
    fake_store, adapter, boundaries, monkeypatch
):
    _existing_package(monkeypatch, boundaries)
    source = await _source(fake_store)
    files = AsyncMock(side_effect=OSError("workspace unavailable"))
    monkeypatch.setattr(publish, "read_user_skill_files", files)

    first = await _run()

    assert first.dispositions == {"provider_error": 1}
    assert (await fake_store.get_source(USER, source.id)).has_unprocessed_revision
    boundaries["write"].assert_not_awaited()

    files.side_effect = None
    files.return_value = []
    recovered = await _run()

    assert recovered.dispositions == {"needs_decision": 1}
    assert not (await fake_store.get_source(USER, source.id)).has_unprocessed_revision
    assert boundaries["review"].await_count == 1
