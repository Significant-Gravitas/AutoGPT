from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import prisma.errors
import prisma.models
import pytest

from backend.api.features.experts import raise_attachments
from backend.api.features.experts.models import RaiseAttachment
from backend.util.exceptions import NotFoundError


def _transaction_losing_the_insert():
    tx = SimpleNamespace(
        expert=SimpleNamespace(
            find_first=AsyncMock(return_value=SimpleNamespace(id="expert-1"))
        ),
        expertworkflow=SimpleNamespace(
            create=AsyncMock(side_effect=prisma.errors.UniqueViolationError({}))
        ),
    )

    @asynccontextmanager
    async def fake_transaction(*args, **kwargs):
        yield tx

    return fake_transaction


@asynccontextmanager
async def _marketplace_install_race(existing: object | None):
    """Marketplace install where a concurrent raise already made the row."""
    workflow_client = SimpleNamespace(find_first=AsyncMock(return_value=existing))
    with (
        patch.object(
            raise_attachments, "transaction", _transaction_losing_the_insert()
        ),
        patch.object(
            raise_attachments.library_db,
            "is_store_listing_version_available_for_install",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch.object(
            raise_attachments.library_db,
            "add_store_agent_to_library_in_transaction",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(id="library-agent-1"),
        ),
        patch.object(
            prisma.models.ExpertWorkflow, "prisma", return_value=workflow_client
        ),
    ):
        yield workflow_client


async def test_install_marketplace_workflow_accepts_concurrent_winner():
    winner = SimpleNamespace(id="workflow-1")

    async with _marketplace_install_race(winner) as workflow_client:
        await raise_attachments.install_marketplace_workflow(
            "user-1", "expert-1", "slv-1"
        )

    workflow_client.find_first.assert_awaited_once()


async def test_install_marketplace_workflow_reraises_race_without_winner():
    async with _marketplace_install_race(None) as workflow_client:
        with pytest.raises(prisma.errors.UniqueViolationError):
            await raise_attachments.install_marketplace_workflow(
                "user-1", "expert-1", "slv-1"
            )

    workflow_client.find_first.assert_awaited_once()


def _hub_skill(slug: str) -> raise_attachments.ResolvedSkill:
    return raise_attachments.ResolvedSkill(
        attachment=RaiseAttachment(kind="skill", source="marketplace", id=slug),
        name=slug,
        marketplace_slug=slug,
    )


async def test_install_marketplace_skills_reports_a_withdrawn_listing():
    """The real race: the listing passed resolve_attachments and was pulled
    before the install, which is `unavailable` rather than a failed install."""
    with patch.object(
        raise_attachments.skill_db,
        "install_marketplace_skill",
        new_callable=AsyncMock,
        side_effect=NotFoundError("gone"),
    ):
        failures = await raise_attachments.install_marketplace_skills(
            "user-1", "expert-1", [_hub_skill("seo-playbook")]
        )

    assert [(f.kind, f.source, f.id, f.reason) for f in failures] == [
        ("skill", "marketplace", "seo-playbook", "unavailable")
    ]


async def test_install_marketplace_skills_reports_a_broken_install():
    with patch.object(
        raise_attachments.skill_db,
        "install_marketplace_skill",
        new_callable=AsyncMock,
        side_effect=RuntimeError("disk full"),
    ):
        failures = await raise_attachments.install_marketplace_skills(
            "user-1", "expert-1", [_hub_skill("seo-playbook")]
        )

    assert [f.reason for f in failures] == ["installation_failed"]


async def test_install_marketplace_skills_installs_each_slug_and_skips_library():
    library = raise_attachments.ResolvedSkill(
        attachment=RaiseAttachment(kind="skill", source="library", id="my-skill"),
        name="my-skill",
    )
    with patch.object(
        raise_attachments.skill_db,
        "install_marketplace_skill",
        new_callable=AsyncMock,
    ) as install:
        failures = await raise_attachments.install_marketplace_skills(
            "user-1", "expert-1", [_hub_skill("seo-playbook"), library]
        )

    assert failures == []
    install.assert_awaited_once_with("user-1", "seo-playbook", expert_id="expert-1")


async def test_resolve_skill_addresses_a_hub_listing_by_slug():
    """The name on the row has to be the slug the install names the copy
    after, or the folder and Expert.skills disagree."""
    attachment = RaiseAttachment(
        kind="skill", source="marketplace", id=" SEO-Playbook "
    )
    with patch.object(
        raise_attachments.skill_db,
        "get_marketplace_skill",
        new_callable=AsyncMock,
    ) as get_skill:
        resolved = await raise_attachments.resolve_attachments("user-1", [attachment])

    get_skill.assert_awaited_once_with("seo-playbook")
    assert [(s.name, s.marketplace_slug) for s in resolved.skills] == [
        ("seo-playbook", "seo-playbook")
    ]
    assert resolved.library_skill_names == []


async def test_resolve_skill_rejects_a_listing_that_is_not_there():
    attachment = RaiseAttachment(kind="skill", source="marketplace", id="ghost")
    with patch.object(
        raise_attachments.skill_db,
        "get_marketplace_skill",
        new_callable=AsyncMock,
        side_effect=NotFoundError("no listing"),
    ):
        with pytest.raises(raise_attachments.RaiseAttachmentUnavailableError):
            await raise_attachments.resolve_attachments("user-1", [attachment])
