"""AgentMail managed credential provider.

Uses the org-level AgentMail API key to create a per-user pod and a
pod-scoped API key.  The pod key is stored as an ``is_managed``
credential so it appears automatically in block credential dropdowns.
"""

from __future__ import annotations

import logging

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, Credentials
from backend.integrations.managed_credentials import ManagedCredentialProvider
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class AgentMailManagedProvider(ManagedCredentialProvider):
    provider_name = "agent_mail"

    async def is_available(self) -> bool:
        return bool(settings.secrets.agentmail_api_key)

    async def provision(self, user_id: str) -> Credentials:
        from agentmail import AsyncAgentMail

        client = AsyncAgentMail(api_key=settings.secrets.agentmail_api_key)

        # client_id makes pod creation idempotent — if a pod already exists
        # for this user_id the SDK returns the existing pod.
        pod = await client.pods.create(client_id=user_id, name=f"{user_id}-pod")

        # NOTE: api_keys.create() is NOT idempotent.  If the caller retries
        # after a partial failure (pod created, key created, but store write
        # failed), a second key will be created and the first becomes orphaned
        # on AgentMail's side.  The double-check pattern in _ensure_one
        # (has_managed_credential under lock) prevents this in normal flow;
        # only a crash between key creation and store write can cause it.
        api_key_obj = await client.pods.api_keys.create(
            pod_id=pod.pod_id, name=f"{user_id}-agpt-managed"
        )

        return APIKeyCredentials(
            provider=self.provider_name,
            title="AgentMail (managed by AutoGPT)",
            api_key=SecretStr(api_key_obj.api_key),
            expires_at=None,
            is_managed=True,
            metadata={"pod_id": pod.pod_id},
        )

    async def deprovision(self, user_id: str, credential: Credentials) -> None:
        from agentmail import AsyncAgentMail

        pod_id = credential.metadata.get("pod_id")
        if not pod_id:
            logger.warning(
                "Managed credential for user %s has no pod_id in metadata — "
                "skipping AgentMail cleanup",
                user_id,
            )
            return

        client = AsyncAgentMail(api_key=settings.secrets.agentmail_api_key)
        try:
            # Verify the pod actually belongs to this user before deleting,
            # as a safety measure against cross-user deletion via the
            # org-level API key.
            pod = await client.pods.get(pod_id=pod_id)
            if getattr(pod, "client_id", None) and pod.client_id != user_id:
                logger.error(
                    "Pod %s client_id=%s does not match user %s — "
                    "refusing to delete",
                    pod_id,
                    pod.client_id,
                    user_id,
                )
                return
            await client.pods.delete(pod_id=pod_id)
        except Exception:
            logger.warning(
                "Failed to delete AgentMail pod %s for user %s",
                pod_id,
                user_id,
                exc_info=True,
            )
