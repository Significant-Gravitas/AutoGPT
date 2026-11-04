from unittest.mock import AsyncMock

import pytest

from backend.api.features.store.exceptions import StorageUploadError, VirusScanError
from backend.copilot.tools.skills import (
    MAX_DESCRIPTION_CHARS,
    MAX_TRIGGERS,
    SKILL_ORIGIN_USER,
    ParsedSkill,
    SkillWrite,
    _prepare_skill,
    render_skill_markdown,
)

from . import nightly, nightly_test, publish
from .nightly_test import _completion, _proposal, _run, _source
from .publish_test import BODY_V1, EXPERT, NAME, USER, _seed

adapter = nightly_test.adapter
boundaries = nightly_test.boundaries


def _edited_skill(invalid_field):
    return ParsedSkill(
        name=NAME,
        description=(
            "x" * (MAX_DESCRIPTION_CHARS + 1)
            if invalid_field == "description"
            else "Import a CSV"
        ),
        body=BODY_V1 + "\nHuman note added on disk.\n",
        triggers=(
            tuple(f"trigger {i}" for i in range(MAX_TRIGGERS + 1))
            if invalid_field == "triggers"
            else ("csv",)
        ),
    )


def _review_edit(monkeypatch, boundaries, invalid_field=None):
    edited = _edited_skill(invalid_field)
    monkeypatch.setattr(nightly, "_existing_skills", AsyncMock(return_value=[edited]))
    monkeypatch.setattr(
        publish,
        "read_user_skill_markdown",
        AsyncMock(return_value=render_skill_markdown(edited)),
    )
    boundaries["review"].return_value = _completion(_proposal(decision="update"))
    boundaries["write"].side_effect = _validate_registry_write


async def _validate_registry_write(
    user_id, *, name, description, body, triggers=None, **kwargs
):
    _prepare_skill(
        SkillWrite(name=name, description=description, body=body, triggers=triggers),
        SKILL_ORIGIN_USER,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_field", ["description", "triggers"])
async def test_invalid_untracked_edit_settles_after_one_review(
    fake_store, adapter, boundaries, monkeypatch, invalid_field
):
    await _seed(fake_store)
    source = await _source(fake_store)
    _review_edit(monkeypatch, boundaries, invalid_field)
    first = await _run()
    assert first.dispositions == {"skipped": 1}
    ledger = await fake_store.get_review_for_revision(
        USER, source.id, source.revision, 1
    )
    assert ledger.disposition == "skipped" and ledger.cost_microdollars == 1000
    assert (
        await fake_store.get_source(USER, source.id)
    ).processed_revision == source.revision
    assert len(await fake_store.list_versions(USER, EXPERT, NAME)) == 2
    assert (await _run()).skip_reason == "no_eligible_work"
    boundaries["review"].assert_awaited_once()
    boundaries["write"].assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        OSError("workspace unavailable"),
        VirusScanError("scanner unavailable"),
        StorageUploadError("upload unavailable"),
        ValueError("workspace storage quota exceeded"),
    ],
    ids=["io", "scanner", "storage", "quota"],
)
async def test_transient_edit_write_failure_keeps_source_retryable(
    fake_store, adapter, boundaries, monkeypatch, error
):
    await _seed(fake_store)
    source = await _source(fake_store)
    _review_edit(monkeypatch, boundaries)
    boundaries["write"].side_effect = error
    failed = await _run()
    assert failed.dispositions == {"provider_error": 1}
    assert (await fake_store.get_source(USER, source.id)).processed_revision is None
    boundaries["write"].side_effect = _validate_registry_write
    retried = await _run()
    assert retried.dispositions == {"conflict": 1}
    assert boundaries["review"].await_count == 1
    assert boundaries["write"].await_count == 2
    ledger = await fake_store.get_review_for_revision(
        USER, source.id, source.revision, 1
    )
    assert ledger.disposition == "conflict"
    assert (await fake_store.get_source(USER, source.id)).processed_revision is None
