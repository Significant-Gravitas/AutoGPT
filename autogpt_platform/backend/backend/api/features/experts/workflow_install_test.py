"""An expert installing a marketplace workflow is not the user adding a
listing to their library: neither install path may send
``listing_added_to_library``."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import prisma.models
import pytest

from backend.api.features.experts import experts_db


@pytest.mark.asyncio
async def test_hire_preloads_do_not_count_as_listing_adds():
    preload = prisma.models.ExpertWorkflow.model_construct(
        storeListingVersionId="slv-1", scheduleCron=None, StoreListingVersion=None
    )
    workflow_client = SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(id="workflow-1"))
    )
    add = AsyncMock(return_value=SimpleNamespace(id="library-agent-1"))
    with (
        patch.object(experts_db.library_db, "add_store_agent_to_library", add),
        patch.object(
            prisma.models.ExpertWorkflow, "prisma", return_value=workflow_client
        ),
    ):
        failed = await experts_db._install_preloads("expert-1", "user-1", [preload])

    assert failed == []
    add.assert_awaited_once_with("slv-1", "user-1", track_listing_added=False)


@pytest.mark.asyncio
async def test_marketplace_workflow_install_does_not_count_as_listing_add():
    workflow_client = SimpleNamespace(
        find_first=AsyncMock(return_value=None),
        create=AsyncMock(return_value=SimpleNamespace(id="workflow-1")),
    )
    add = AsyncMock(return_value=SimpleNamespace(id="library-agent-1"))
    with (
        patch.object(experts_db.library_db, "add_store_agent_to_library", add),
        patch.object(
            prisma.models.ExpertWorkflow, "prisma", return_value=workflow_client
        ),
        patch.object(experts_db, "_to_workflow_ref", MagicMock()),
        patch.object(experts_db, "emit_funnel_event") as emit,
    ):
        await experts_db._install_marketplace_workflow("user-1", "expert-1", "slv-1")

    add.assert_awaited_once_with("slv-1", "user-1", track_listing_added=False)
    assert emit.call_args.args[1] == "workflow_installed_on_expert"
