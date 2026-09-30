from collections.abc import Awaitable, Callable

import pytest

from backend.blocks.conductor._api import API_V0, ConductorClient
from backend.blocks.conductor.test_fixtures import FakeResponse, client_with


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,args,path",
    [
        (
            ConductorClient.workspace_lifecycle,
            ("ws_1", "sleep"),
            "workspaces/ws_1/sleep",
        ),
        (
            ConductorClient.workspace_lifecycle,
            ("ws_1", "archive"),
            "workspaces/ws_1/archive",
        ),
        (
            ConductorClient.workspace_lifecycle,
            ("ws_1", "unarchive"),
            "workspaces/ws_1/unarchive",
        ),
        (ConductorClient.cancel_session, ("s1",), "sessions/s1/cancel"),
        (ConductorClient.archive_session, ("s1",), "sessions/s1/archive"),
        (ConductorClient.rotate_routine_secret, ("r1",), "routines/r1/rotate-secret"),
    ],
)
async def test_action_posts_send_json_body(
    operation: Callable[..., Awaitable[dict]], args: tuple[str, ...], path: str
):
    client, request = client_with(FakeResponse(200, {"status": "ok"}))

    await operation(client, *args)

    request.assert_awaited_once_with("POST", f"{API_V0}/{path}", json={}, params=None)
