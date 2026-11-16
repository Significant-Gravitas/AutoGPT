import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from prisma.models import TransferRequest

from backend.api.features.transfers import db as transfers_db


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("terminal_status", ["REJECTED", "CANCELLED", "COMPLETED"])
async def test_stale_approval_cannot_reopen_terminal_transfer(
    transfer_world, terminal_status
):
    world = transfer_world
    transfer = await transfers_db.create_transfer(
        world["org_a"],
        world["org_b"],
        world["resource_type"],
        world["resource_id"],
        world["source_user"],
    )
    await transfers_db.approve_transfer(
        transfer.id, world["source_user"], world["org_a"]
    )
    stale = await TransferRequest.prisma().find_unique_or_raise(
        where={"id": transfer.id}
    )
    if terminal_status == "COMPLETED":
        await transfers_db.approve_transfer(
            transfer.id, world["target_user"], world["org_b"]
        )
        await transfers_db.execute_transfer(
            transfer.id, world["source_user"], world["org_a"]
        )
    else:
        await TransferRequest.prisma().update(
            where={"id": transfer.id}, data={"status": terminal_status}
        )
    before = await TransferRequest.prisma().find_unique_or_raise(
        where={"id": transfer.id}
    )

    with patch.object(
        type(transfers_db.prisma.transferrequest),
        "find_unique",
        AsyncMock(return_value=stale),
    ), pytest.raises(ValueError):
        await transfers_db.approve_transfer(
            transfer.id, world["target_user"], world["org_b"]
        )

    after = await TransferRequest.prisma().find_unique_or_raise(
        where={"id": transfer.id}
    )
    assert after == before


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("same_side", [True, False])
async def test_concurrent_approvals_preserve_both_sides(transfer_world, same_side):
    world = transfer_world
    transfer = await transfers_db.create_transfer(
        world["org_a"],
        world["org_b"],
        world["resource_type"],
        world["resource_id"],
        world["source_user"],
    )
    results = await asyncio.gather(
        transfers_db.approve_transfer(
            transfer.id, world["source_user"], world["org_a"]
        ),
        transfers_db.approve_transfer(
            transfer.id,
            world["source_user"] if same_side else world["target_user"],
            world["org_a"] if same_side else world["org_b"],
        ),
        return_exceptions=True,
    )
    errors = [r for r in results if isinstance(r, BaseException)]
    assert len(errors) == int(same_side), results
    assert all(isinstance(error, ValueError) for error in errors)
    stored = await TransferRequest.prisma().find_unique_or_raise(
        where={"id": transfer.id}
    )
    assert stored.sourceApprovedByUserId == world["source_user"]
    assert stored.targetApprovedByUserId == (
        None if same_side else world["target_user"]
    )
    if not same_side:
        result = await transfers_db.execute_transfer(
            transfer.id, world["source_user"], world["org_a"]
        )
        assert result.status == "COMPLETED"
