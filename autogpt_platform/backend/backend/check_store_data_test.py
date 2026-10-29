from unittest.mock import AsyncMock

import pytest
from prisma import Prisma
from prisma.errors import ClientNotConnectedError

from backend.check_store_data import check_store_data
from backend.data.db import prisma


@pytest.mark.asyncio
async def test_raw_store_queries_use_the_supplied_client(mocker) -> None:
    client = Prisma()
    mocker.patch.object(
        type(client.storelisting), "find_many", new=AsyncMock(return_value=[])
    )
    mocker.patch.object(
        type(client.storelistingversion), "find_many", new=AsyncMock(return_value=[])
    )
    mocker.patch.object(
        type(client.storelistingreview), "find_many", new=AsyncMock(return_value=[])
    )
    assert client is not prisma

    async def query_raw(queried_client: Prisma, *args, **kwargs):
        if queried_client is not client:
            raise ClientNotConnectedError()
        return []

    query = mocker.patch.object(
        Prisma, "query_raw", autospec=True, side_effect=query_raw
    )

    await check_store_data(client)

    assert query.await_count == 4
