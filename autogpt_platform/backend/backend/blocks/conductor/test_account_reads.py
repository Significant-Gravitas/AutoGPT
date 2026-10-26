import asyncio
from functools import partial
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.blocks.conductor.account import ConductorGetAccountBlock
from backend.blocks.conductor.test_fixtures import TEST_CREDENTIALS


@pytest.mark.asyncio
async def test_account_reads_start_concurrently():
    started: set[str] = set()
    all_started = asyncio.Event()

    async def read(name: str, *args):
        started.add(name)
        if len(started) == 4:
            all_started.set()
        await all_started.wait()
        return {"userId": "u1"} if name == "user" else {"data": [{"id": name}]}

    client = Mock(
        get_me=AsyncMock(side_effect=partial(read, "user")),
        list_projects=AsyncMock(side_effect=partial(read, "projects")),
        list_sections=AsyncMock(side_effect=partial(read, "sections")),
        list_routines=AsyncMock(side_effect=partial(read, "routines")),
    )
    with patch("backend.blocks.conductor.account.ConductorClient", return_value=client):
        result = await asyncio.wait_for(
            ConductorGetAccountBlock()._fetch(TEST_CREDENTIALS, 100), timeout=1
        )

    assert result == {
        "user": {"userId": "u1"},
        "projects": [{"id": "projects"}],
        "sections": [{"id": "sections"}],
        "routines": [{"id": "routines"}],
    }
