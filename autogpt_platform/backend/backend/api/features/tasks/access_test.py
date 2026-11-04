from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import create_model

from backend.api.features.tasks.access import _RemoteTasksStore
from backend.api.features.tasks.models import Actor, TaskUpdate


async def test_a_remote_update_sends_only_the_fields_it_sets():
    client = MagicMock()
    client.tasks_update_task = AsyncMock()
    with patch(
        "backend.api.features.tasks.access.get_database_manager_async_client",
        return_value=client,
    ):
        await _RemoteTasksStore().update_task(
            "user-1", "task-1", TaskUpdate(title="Hire 2 experts"), Actor.system()
        )

    sent = client.tasks_update_task.await_args.args[2]
    assert sent == {"title": "Hire 2 experts"}

    body = create_model("Body", data=(TaskUpdate, ...))
    received = body.model_validate({"data": sent}).data
    assert received.model_fields_set == {"title"}


def test_other_remote_calls_pass_straight_through():
    client = MagicMock()
    with patch(
        "backend.api.features.tasks.access.get_database_manager_async_client",
        return_value=client,
    ):
        assert _RemoteTasksStore().get_task is client.tasks_get_task
