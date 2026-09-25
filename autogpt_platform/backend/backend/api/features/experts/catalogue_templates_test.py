"""Real PostgreSQL preservation tests. Opt in with a dedicated local test URL."""

import asyncio
import os
from urllib.parse import urlparse
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma import Prisma

from backend.api.features.experts import catalogue_templates as provisioning
from backend.api.features.experts.catalogue_template_test_helpers import Fixture

pytestmark = pytest.mark.catalogue_isolated


@pytest.fixture(scope="session")
def graph_cleanup():
    """These tests own their Prisma connection and cleanup, not SpinTestServer."""
    yield


@pytest_asyncio.fixture
async def fixture():
    url = os.environ.get("CATALOGUE_TEST_DATABASE_URL")
    if not url:
        pytest.skip(
            "set CATALOGUE_TEST_DATABASE_URL to a dedicated local PostgreSQL database"
        )
    if urlparse(url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        pytest.fail(
            "these tests are restricted to a dedicated local PostgreSQL database"
        )
    db = Prisma(datasource={"url": url})
    await db.connect()
    helper = Fixture(db)
    try:
        yield helper
    finally:
        await helper.cleanup()
        await db.disconnect()


def adoption(definitions, **existing):
    return provisioning.TemplateAdoption(
        experts={key: existing.get(key) for key in definitions}
    )


async def apply(db, mapping):
    preview = await provisioning.preview_templates(db, mapping)
    return await provisioning.apply_templates(
        db, mapping, expected_preview_sha256=preview.preview_sha256
    )


@pytest.mark.asyncio
async def test_creates_archived_template_with_disabled_routines_and_official_preloads(
    fixture, monkeypatch
):
    definitions = fixture.definitions(monkeypatch)
    key, definition = next(iter(definitions.items()))
    slug = f"workflow-{uuid4().hex}"
    await fixture.workflow(slug, official=False)
    official_version = await fixture.workflow(slug, official=True)
    definition["preloads"] = [{"slug": slug, "cron": None}]
    result = await apply(fixture.db, adoption(definitions))
    record = await fixture.db.expert.find_unique(where={"id": result.experts[key]})
    assert record and record.isArchived and record.isTemplate
    assert record.ownerUserId is record.organizationId is record.teamId is None
    assert record.identity == "Reviewed persona"
    assert result.activate_experts == [key]
    routines = await fixture.db.expertroutine.find_many(where={"expertId": record.id})
    assert len(routines) == 1
    assert routines[0].source == "TEMPLATE" and not routines[0].grantsCredentials
    assert routines[0].enabledAt is None and routines[0].scheduleIds == []
    workflows = await fixture.db.expertworkflow.find_many(where={"expertId": record.id})
    assert (
        len(workflows) == 1 and workflows[0].storeListingVersionId == official_version
    )
    assert workflows[0].libraryAgentId is workflows[0].scheduleId is None
    assert await fixture.db.expertskilllisting.count(where={"expertId": record.id}) == 0
    assert await fixture.db.expertcredential.count(where={"expertId": record.id}) == 0


@pytest.mark.asyncio
async def test_preserves_existing_template_customer_hire_and_custom_expert(
    fixture, monkeypatch
):
    definitions = fixture.definitions(monkeypatch, count=2)
    existing_key, missing_key = definitions
    existing = await fixture.expert(
        definitions[existing_key]["fields"]["name"], isTemplate=True, isArchived=True
    )
    owner = await fixture.user()
    hire = await fixture.expert(
        existing.name, ownerUserId=owner, sourceTemplateId=existing.id
    )
    custom = await fixture.expert(
        definitions[missing_key]["fields"]["name"], ownerUserId=owner
    )
    for record in (existing, hire, custom):
        await fixture.db.expertroutine.create(
            data={
                "expertId": record.id,
                "title": "Personal routine",
                "prompt": "Keep me",
                "crons": [],
                "asks": [],
                "scheduleIds": ["existing-schedule"],
                "grantsCredentials": True,
            }
        )
        await fixture.db.expertcredential.create(
            data={
                "expertId": record.id,
                "credentialId": f"personal-{record.id}",
                "provider": "google",
            }
        )
    ids = [existing.id, hire.id, custom.id]
    before = await fixture.fingerprint(ids)
    result = await apply(
        fixture.db, adoption(definitions, **{existing_key: existing.id})
    )
    assert await fixture.fingerprint(ids) == before
    assert result.experts[existing_key] == existing.id
    assert result.activate_experts == [missing_key]


@pytest.mark.asyncio
@pytest.mark.parametrize("ownership", ["owner", "organization", "not-template"])
async def test_rejects_owned_or_non_template_adoption(fixture, monkeypatch, ownership):
    definitions = fixture.definitions(monkeypatch)
    key = next(iter(definitions))
    kwargs = {"isTemplate": ownership != "not-template"}
    if ownership == "owner":
        kwargs["ownerUserId"] = await fixture.user()
    if ownership == "organization":
        kwargs["organizationId"] = str(uuid4())
    record = await fixture.expert(definitions[key]["fields"]["name"], **kwargs)
    before = await fixture.fingerprint([record.id])
    with pytest.raises(ValueError, match="unowned platform template"):
        await apply(fixture.db, adoption(definitions, **{key: record.id}))
    assert await fixture.fingerprint([record.id]) == before


@pytest.mark.asyncio
async def test_expected_absence_and_duplicate_names_are_fail_closed(
    fixture, monkeypatch
):
    definitions = fixture.definitions(monkeypatch)
    key = next(iter(definitions))
    first = await fixture.expert(definitions[key]["fields"]["name"], isTemplate=True)
    with pytest.raises(ValueError, match="unexpected existing template"):
        await apply(fixture.db, adoption(definitions))
    await fixture.expert(first.name, isTemplate=True)
    with pytest.raises(ValueError, match="unexpected existing template"):
        await apply(fixture.db, adoption(definitions, **{key: first.id}))


@pytest.mark.asyncio
async def test_existing_row_drift_rejects_preview_without_creating_missing_template(
    fixture, monkeypatch
):
    definitions = fixture.definitions(monkeypatch, count=2)
    key, missing = definitions
    record = await fixture.expert(definitions[key]["fields"]["name"], isTemplate=True)
    mapping = adoption(definitions, **{key: record.id})
    preview = await provisioning.preview_templates(fixture.db, mapping)
    await fixture.db.expert.update(
        where={"id": record.id}, data={"identity": "Later edit"}
    )
    with pytest.raises(ValueError, match="stale"):
        await provisioning.apply_templates(
            fixture.db, mapping, expected_preview_sha256=preview.preview_sha256
        )
    assert (
        await fixture.db.expert.count(
            where={"name": definitions[missing]["fields"]["name"]}
        )
        == 0
    )


@pytest.mark.asyncio
async def test_definition_drift_rejects_preview(fixture, monkeypatch):
    definitions = fixture.definitions(monkeypatch)
    mapping = adoption(definitions)
    preview = await provisioning.preview_templates(fixture.db, mapping)
    next(iter(definitions.values()))["fields"]["identity"] = "Different source"
    with pytest.raises(ValueError, match="stale"):
        await provisioning.apply_templates(
            fixture.db, mapping, expected_preview_sha256=preview.preview_sha256
        )


@pytest.mark.asyncio
async def test_other_creator_slug_cannot_replace_missing_official_workflow(
    fixture, monkeypatch
):
    definitions = fixture.definitions(monkeypatch)
    definition = next(iter(definitions.values()))
    slug = f"workflow-{uuid4().hex}"
    await fixture.workflow(slug, official=False)
    definition["preloads"] = [{"slug": slug, "cron": None}]
    with pytest.raises(ValueError, match="official autogpt preload"):
        await apply(fixture.db, adoption(definitions))
    assert (
        await fixture.db.expert.count(where={"name": definition["fields"]["name"]}) == 0
    )


@pytest.mark.asyncio
async def test_failure_rolls_back_all_new_templates(fixture, monkeypatch):
    definitions = fixture.definitions(monkeypatch, count=2)
    create = provisioning._create_template
    calls = 0

    async def failing_create(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected failure")
        await create(*args)

    monkeypatch.setattr(provisioning, "_create_template", failing_create)
    with pytest.raises(RuntimeError, match="injected failure"):
        await apply(fixture.db, adoption(definitions))
    assert await fixture.db.expert.count(where={"name": {"in": fixture.names}}) == 0


@pytest.mark.asyncio
async def test_readopting_new_ids_is_noop_and_old_missing_mapping_cannot_duplicate(
    fixture, monkeypatch
):
    definitions = fixture.definitions(monkeypatch)
    mapping = adoption(definitions)
    result = await apply(fixture.db, mapping)
    before = await fixture.fingerprint(list(result.experts.values()))
    with pytest.raises(ValueError, match="unexpected existing template"):
        await apply(fixture.db, mapping)
    repeated = await apply(
        fixture.db, provisioning.TemplateAdoption(experts=result.experts)
    )
    assert repeated.activate_experts == []
    assert await fixture.fingerprint(list(result.experts.values())) == before


@pytest.mark.asyncio
async def test_concurrent_apply_cannot_create_duplicate_templates(fixture, monkeypatch):
    definitions = fixture.definitions(monkeypatch)
    mapping = adoption(definitions)
    preview = await provisioning.preview_templates(fixture.db, mapping)
    results = await asyncio.gather(
        *[
            provisioning.apply_templates(
                fixture.db, mapping, expected_preview_sha256=preview.preview_sha256
            )
            for _ in range(2)
        ],
        return_exceptions=True,
    )
    assert (
        sum(isinstance(result, provisioning.ProvisionedTemplates) for result in results)
        == 1
    )
    assert sum(isinstance(result, ValueError) for result in results) == 1
    assert await fixture.db.expert.count(where={"name": {"in": fixture.names}}) == 1


def test_duplicate_adoption_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        provisioning.TemplateAdoption(experts={"first": "one-id", "second": "one-id"})
