from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import prisma.errors
import prisma.models
import pytest

from backend.api.features.experts import raise_attachments


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
