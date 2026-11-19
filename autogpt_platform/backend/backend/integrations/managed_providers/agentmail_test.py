"""The AgentMail pod key the platform provisions is billed like a platform key."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.model import is_platform_funded_managed
from backend.integrations.managed_providers.agentmail import AgentMailManagedProvider


@pytest.mark.asyncio
async def test_provisioned_key_is_marked_platform_funded():
    client = MagicMock()
    client.pods.create = AsyncMock(return_value=SimpleNamespace(pod_id="pod-1"))
    client.pods.api_keys.create = AsyncMock(
        return_value=SimpleNamespace(api_key="pod-key")
    )

    with patch("agentmail.AsyncAgentMail", return_value=client):
        credential = await AgentMailManagedProvider().provision("user-1", MagicMock())

    assert credential.is_managed
    assert credential.provider == "agent_mail"
    assert is_platform_funded_managed(credential.id)


def test_only_minted_ids_are_platform_funded():
    assert not is_platform_funded_managed("user-own-key")
    assert not is_platform_funded_managed("agent_mail-default")
    assert not is_platform_funded_managed("ayrshare-managed-1")
