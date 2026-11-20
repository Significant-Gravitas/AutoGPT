"""Deleting a run ends its share; only the owner's run is touched.

End-to-end coverage through the public routes lives in
``backend/api/features/graph_executions/deleted_share_test.py``.
"""

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.sharing.execution_shares import soft_delete_and_unshare_execution


@pytest.fixture
def prisma_calls() -> Iterator[dict[str, MagicMock]]:
    module = "backend.data.sharing.execution_shares"
    with (
        patch(f"{module}.AgentGraphExecution.prisma") as execution,
        patch(f"{module}.SharedExecutionFile.prisma") as files,
        patch(f"{module}.ChatLinkedShare.prisma") as links,
        patch(f"{module}.transaction") as transaction,
    ):
        transaction.return_value.__aenter__ = AsyncMock(return_value="tx")
        transaction.return_value.__aexit__ = AsyncMock(return_value=False)
        files.return_value.delete_many = AsyncMock(return_value=1)
        links.return_value.delete_many = AsyncMock(return_value=1)
        yield {"execution": execution, "files": files, "links": links}


@pytest.mark.asyncio
async def test_soft_delete_clears_the_share_and_its_rows(prisma_calls):
    update_many = AsyncMock(return_value=1)
    prisma_calls["execution"].return_value.update_many = update_many

    assert await soft_delete_and_unshare_execution("exec-1", "user-1") == 1

    update_many.assert_awaited_once_with(
        where={"id": "exec-1", "userId": "user-1"},
        data={
            "isDeleted": True,
            "isShared": False,
            "shareToken": None,
            "sharedAt": None,
            "sharedVia": None,
        },
    )
    for table in ("files", "links"):
        prisma_calls[table].assert_called_with("tx")
        prisma_calls[table].return_value.delete_many.assert_awaited_once_with(
            where={"executionId": "exec-1"}
        )


@pytest.mark.asyncio
async def test_soft_delete_of_someone_elses_run_touches_no_share_rows(prisma_calls):
    prisma_calls["execution"].return_value.update_many = AsyncMock(return_value=0)

    assert await soft_delete_and_unshare_execution("exec-1", "intruder") == 0

    prisma_calls["files"].return_value.delete_many.assert_not_awaited()
    prisma_calls["links"].return_value.delete_many.assert_not_awaited()
