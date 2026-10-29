import hmac
import logging

from fastapi import HTTPException, Request
from pydantic import SecretStr
from strenum import StrEnum

from backend.data.integrations import WebhookWithRelations
from backend.sdk import Credentials, ManualWebhookManagerBase, Webhook

logger = logging.getLogger(__name__)


class GenericWebhookType(StrEnum):
    PLAIN = "plain"


class GenericWebhooksManager(ManualWebhookManagerBase):
    WebhookType = GenericWebhookType

    # Name of the input field on `GenericWebhookTriggerBlock` that carries the
    # optional user-chosen secret. Read here at verification time rather than
    # snapshotted at webhook registration so a later change to the secret on
    # the block takes effect immediately.
    SECRET_TOKEN_INPUT = "secret_token"
    SECRET_HEADER = "X-Webhook-Secret"

    @classmethod
    async def verify_signature(
        cls, webhook: WebhookWithRelations, request: Request
    ) -> None:
        # Find any non-empty `secret_token` configured on a triggered node or
        # preset attached to this webhook. The webhook is loaded via
        # `get_webhook(..., include_relations=True)` in the router so these
        # relations are already populated.
        expected = cls._configured_secret(webhook)
        if not expected:
            # No secret configured — back-compat with unauthenticated generic
            # webhooks. Webhook URL is the only credential.
            return

        provided = request.headers.get(cls.SECRET_HEADER)
        # constant-time compare to prevent timing side-channel
        if not provided or not hmac.compare_digest(provided, expected):
            raise HTTPException(
                status_code=403,
                detail=f"Invalid or missing {cls.SECRET_HEADER} header",
            )

    @classmethod
    def _configured_secret(cls, webhook: WebhookWithRelations) -> str | None:
        sources = [node.input_default for node in webhook.triggered_nodes] + [
            preset.inputs for preset in webhook.triggered_presets
        ]

        found: list[str] = []
        for src in sources:
            value = src.get(cls.SECRET_TOKEN_INPUT)
            # Stored values may arrive as plain strings or as SecretStr
            # depending on serialization path; normalize.
            if isinstance(value, SecretStr):
                value = value.get_secret_value()
            if isinstance(value, str) and value.strip():
                found.append(value)

        if not found:
            return None
        # Compute the distinct-count before logging so the secret values
        # themselves don't flow into the logger call args (CodeQL's
        # clear-text-logging taint analysis flags any expression that
        # derives from a secret-typed variable, even just its length).
        distinct_count = len(set(found))
        if distinct_count > 1:
            # Multiple attached targets configured different tokens. We only
            # have one HMAC comparison to make, so log loudly — the first
            # token wins but the user almost certainly didn't intend this.
            logger.warning(
                "Webhook %s has %d distinct secret_token values across "
                "attached targets; using the first one. All targets attached "
                "to the same webhook must share the same secret.",
                webhook.id,
                distinct_count,
            )
        return found[0]

    @classmethod
    async def validate_payload(
        cls, webhook: Webhook, request: Request, credentials: Credentials | None = None
    ) -> tuple[dict, str]:
        payload = await request.json()
        event_type = GenericWebhookType.PLAIN

        return payload, event_type
