import hmac
import logging

from fastapi import HTTPException, Request
from strenum import StrEnum

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
    async def verify_signature(cls, webhook: Webhook, request: Request) -> None:
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
        if not provided or not hmac.compare_digest(provided, expected):
            raise HTTPException(
                status_code=403,
                detail=f"Invalid or missing {cls.SECRET_HEADER} header",
            )

    @classmethod
    def _configured_secret(cls, webhook: Webhook) -> str | None:
        sources: list[dict] = []
        # WebhookWithRelations exposes triggered_nodes/triggered_presets; on a
        # plain `Webhook` they're absent.
        for node in getattr(webhook, "triggered_nodes", []) or []:
            sources.append(getattr(node, "input_default", {}) or {})
        for preset in getattr(webhook, "triggered_presets", []) or []:
            sources.append(getattr(preset, "inputs", {}) or {})

        for src in sources:
            value = src.get(cls.SECRET_TOKEN_INPUT)
            if value:
                # Stored values may arrive as plain strings or as SecretStr
                # depending on serialization path; normalize.
                if hasattr(value, "get_secret_value"):
                    value = value.get_secret_value()
                if isinstance(value, str) and value.strip():
                    return value
        return None

    @classmethod
    async def validate_payload(
        cls, webhook: Webhook, request: Request, credentials: Credentials | None = None
    ) -> tuple[dict, str]:
        payload = await request.json()
        event_type = GenericWebhookType.PLAIN

        return payload, event_type
