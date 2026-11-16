import asyncio
import uuid

import pytest

from backend.api.features.library.exceptions import FolderAlreadyExistsError
from backend.data.skill_learning import LearningAccessError
from backend.util.clients import get_database_manager_async_client
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_a_refusal_defined_outside_util_exceptions_reaches_the_caller_at_once(
    server: SpinTestServer, setup_test_user: str
):
    """Both refusals are plain ``Exception``s defined beside their feature; the
    client must rebuild them, not retry them as 500s."""
    client = get_database_manager_async_client()
    # Untimed: the server may still be starting, and that is not what is measured.
    await client.health_check_async()
    name = f"Duplicate {uuid.uuid4()}"
    await client.create_folder(user_id=setup_test_user, name=name)

    with pytest.raises(FolderAlreadyExistsError):
        await asyncio.wait_for(
            client.create_folder(user_id=setup_test_user, name=name), timeout=20
        )
    with pytest.raises(LearningAccessError):
        await asyncio.wait_for(
            client.require_owned_source(setup_test_user, str(uuid.uuid4())),
            timeout=20,
        )
