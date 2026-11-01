"""Microsoft 365 Copilot turns over Graph's chatOverStream, served by the fake."""

from functools import partial
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from backend.copilot.model import ChatSession
from backend.data.model import OAuth2Credentials
from backend.integrations.credential_lease import CredentialLease
from backend.integrations.microsoft_365_copilot import service as m365
from backend.integrations.microsoft_365_copilot.client import Microsoft365CopilotClient
from backend.integrations.oauth.microsoft_365_copilot import (
    Microsoft365CopilotDeviceAuthHandler,
)
from backend.integrations.providers import ProviderName

from .provider import FakeProvider

MICROSOFT_365_CREDENTIAL_ID = "microsoft-365-credential"


class Microsoft365Turns:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, provider: FakeProvider) -> None:
        monkeypatch.setattr(
            m365,
            "Microsoft365CopilotClient",
            partial(Microsoft365CopilotClient, base_url=f"{provider.url}/graph"),
        )
        monkeypatch.setattr(
            m365, "upsert_chat_session", AsyncMock(side_effect=lambda session: session)
        )
        monkeypatch.setattr(m365, "_get_timezone", AsyncMock(return_value="UTC"))
        credentials = OAuth2Credentials(
            id=MICROSOFT_365_CREDENTIAL_ID,
            provider=ProviderName.MICROSOFT_365_COPILOT,
            access_token=SecretStr("graph-billing-e2e"),
            scopes=list(Microsoft365CopilotDeviceAuthHandler.CHAT_SCOPES),
            title="Microsoft 365 Copilot",
        )
        self._lease = CredentialLease(
            credentials, lock=MagicMock(), checkpoint_callback=AsyncMock()
        )

    async def __call__(
        self, session: ChatSession, prompt: str, *, new_pod: bool
    ) -> None:
        # Graph keeps the conversation; nothing of it lives on a pod.
        async for _ in m365.stream_chat_completion_microsoft_365(
            session.session_id,
            message=prompt,
            is_user_message=True,
            user_id=session.user_id,
            session=session,
            credential_lease=self._lease,
        ):
            pass
