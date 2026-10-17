"""Undo what the account set up at other services through its integrations:
webhooks registered with providers, and managed credentials the platform
provisioned for it (AgentMail pods and the like).

Revoking the OAuth grants themselves is left to the self-service endpoint
(SECRT-2877), which reuses the credential-deletion route's revocation path;
here the tokens are only deleted with the account's rows.
"""

import logging

from prisma.models import IntegrationWebhook

from backend.api.features.integrations.router import remove_all_webhooks_for_credentials
from backend.data.account_deletion.models import StepReport, StepStatus, Subject
from backend.data.user import cleanup_user_managed_credentials
from backend.integrations.credentials_store import IntegrationCredentialsStore

logger = logging.getLogger(__name__)


async def release_integrations(subject: Subject, *, dry_run: bool) -> StepReport:
    """Deregister the account's webhooks at their providers, then deprovision
    its managed credentials. A webhook whose credentials are already gone
    cannot be deregistered; its row is deleted with the account's rows."""
    store = IntegrationCredentialsStore()
    credentials = await store.get_all_creds(subject.user_id)
    webhooks = await IntegrationWebhook.prisma().find_many(
        where={"userId": subject.user_id}
    )
    # The store also lists the platform's own system credentials; only the
    # ones a webhook was registered with matter here.
    used = {webhook.credentialsId for webhook in webhooks}
    managed = sum(1 for c in credentials if c.is_managed)
    if not dry_run:
        for credential in credentials:
            if credential.id in used:
                await remove_all_webhooks_for_credentials(
                    subject.user_id, credential, force=True
                )
        await cleanup_user_managed_credentials(subject.user_id, store)
    return StepReport(
        name="integrations",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts={
            "webhooks": len(webhooks),
            "managed_credentials": managed,
        },
    )
