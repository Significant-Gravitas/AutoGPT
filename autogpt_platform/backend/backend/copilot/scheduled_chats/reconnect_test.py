import asyncio

import pytest

from backend.copilot.scheduled_chats import reconnect


@pytest.mark.asyncio
async def test_a_connect_does_not_wait_for_the_resume(mocker):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_resume(user_id: str, provider: str) -> None:
        started.set()
        await release.wait()

    mocker.patch.object(reconnect, "resume_after_reconnect", side_effect=slow_resume)

    task = reconnect.resume_after_reconnect_in_background("user-1", "discord")
    await started.wait()

    assert not task.done()
    release.set()
    await task
