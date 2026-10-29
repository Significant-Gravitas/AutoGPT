"""
Exa Websets webhook management blocks: create, get, update and delete.

These manage webhook subscriptions directly at Exa, for sending Websets events
to any URL. To start a workflow from Websets events, use the "Exa Webset
Webhook" trigger block instead: it registers and verifies its own webhook.
"""

from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import exa
from ._exa_rest import path_segment
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._webhook import ExaEventType
from ._websets_webhook_samples import (
    SAMPLE_CREATED_WEBHOOK,
    SAMPLE_WEBHOOK,
    SAMPLE_WEBHOOK_ID,
)
from .websets_webhooks_models import (
    WEBHOOKS_URL,
    WebsetsTeamApiMixin,
    WebsetWebhookModel,
    event_type_values,
)

_SAMPLE_MODEL = WebsetWebhookModel.from_api(SAMPLE_WEBHOOK)
_EVENTS_HELP = (
    "Events to deliver (1-19). import.processing is legacy and is rejected; "
    "use import.completed."
)


class ExaCreateWebsetWebhookBlock(WebsetsTeamApiMixin, Block):
    """Subscribe a URL to Websets events."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        url: str = SchemaField(
            description="HTTPS URL Exa should deliver events to",
            placeholder="https://example.com/webhooks/exa",
        )
        events: list[ExaEventType] = SchemaField(
            description=_EVENTS_HELP, min_length=1, max_length=19
        )
        metadata: dict[str, str] = SchemaField(
            default_factory=dict,
            description="Key-value pairs to store with the webhook",
        )

    class Output(BlockSchemaOutput):
        webhook_id: str = SchemaField(description="The new webhook's ID")
        secret: str = SchemaField(
            description="Signing secret for verifying deliveries (Exa-Signature "
            "header). Exa returns it only now, so store it safely."
        )
        webhook: WebsetWebhookModel = SchemaField(
            title="Webhook", description="The full webhook"
        )

    def __init__(self):
        super().__init__(
            id="a28ca1e1-31a3-4f3d-86dc-7ef1042c163a",
            description="Create an Exa Websets webhook that sends the chosen events "
            "to a URL, and get its one-time signing secret",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCreateWebsetWebhookBlock.Input,
            output_schema=ExaCreateWebsetWebhookBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "url": "https://example.com/webhooks/exa",
                "events": [
                    ExaEventType.WEBSET_ITEM_CREATED,
                    ExaEventType.WEBSET_IDLE,
                ],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("webhook_id", SAMPLE_WEBHOOK_ID),
                ("secret", str),
                ("webhook", _SAMPLE_MODEL),
            ],
            test_mock={"call_exa": lambda *args, **kwargs: SAMPLE_CREATED_WEBHOOK},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        body: dict[str, Any] = {
            "url": input_data.url.strip(),
            "events": event_type_values(input_data.events),
        }
        if input_data.metadata:
            body["metadata"] = input_data.metadata
        data = await self.call_exa(credentials, "POST", WEBHOOKS_URL, body=body)
        webhook = WebsetWebhookModel.from_api(data)
        yield "webhook_id", webhook.id
        yield "secret", data.get("secret") or ""
        yield "webhook", webhook


class ExaGetWebsetWebhookBlock(WebsetsTeamApiMixin, Block):
    """Fetch a Websets webhook (its secret is never returned again)."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webhook_id: str = SchemaField(
            description="The webhook to fetch", placeholder=SAMPLE_WEBHOOK_ID
        )

    class Output(BlockSchemaOutput):
        webhook_id: str = SchemaField(description="The webhook's ID")
        status: str = SchemaField(description="active or inactive")
        events: list[str] = SchemaField(description="Subscribed events")
        url: str = SchemaField(description="Where events are delivered")
        webhook: WebsetWebhookModel = SchemaField(
            title="Webhook", description="The full webhook"
        )

    def __init__(self):
        super().__init__(
            id="a66fcc3f-5dee-4680-876b-3b3b8aa5a5c3",
            description="Get an Exa Websets webhook's URL, events, status and metadata",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetWebsetWebhookBlock.Input,
            output_schema=ExaGetWebsetWebhookBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "webhook_id": SAMPLE_WEBHOOK_ID,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("webhook_id", SAMPLE_WEBHOOK_ID),
                ("status", "active"),
                ("events", SAMPLE_WEBHOOK["events"]),
                ("url", SAMPLE_WEBHOOK["url"]),
                ("webhook", _SAMPLE_MODEL),
            ],
            test_mock={"call_exa": lambda *args, **kwargs: SAMPLE_WEBHOOK},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.call_exa(
            credentials, "GET", f"{WEBHOOKS_URL}/{path_segment(input_data.webhook_id)}"
        )
        webhook = WebsetWebhookModel.from_api(data)
        yield "webhook_id", webhook.id
        yield "status", webhook.status
        yield "events", webhook.events
        yield "url", webhook.url
        yield "webhook", webhook


class ExaUpdateWebsetWebhookBlock(WebsetsTeamApiMixin, Block):
    """Change a Websets webhook's URL, events or metadata."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webhook_id: str = SchemaField(
            description="The webhook to update", placeholder=SAMPLE_WEBHOOK_ID
        )
        url: str | None = SchemaField(
            default=None,
            description="New delivery URL; leave empty to keep it",
            advanced=False,
        )
        events: list[ExaEventType] | None = SchemaField(
            default=None,
            description="Replace the subscribed events; leave empty to keep them. "
            + _EVENTS_HELP,
            advanced=False,
        )
        metadata: dict[str, str] | None = SchemaField(
            default=None, description="Replace the metadata; leave empty to keep it"
        )

    class Output(BlockSchemaOutput):
        webhook_id: str = SchemaField(description="The webhook's ID")
        webhook: WebsetWebhookModel = SchemaField(
            title="Webhook", description="The updated webhook"
        )

    def __init__(self):
        super().__init__(
            id="6dd28d43-70ac-4e53-abd3-26131918a005",
            description="Update an Exa Websets webhook's URL, events or metadata",
            categories={BlockCategory.SEARCH},
            input_schema=ExaUpdateWebsetWebhookBlock.Input,
            output_schema=ExaUpdateWebsetWebhookBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "webhook_id": SAMPLE_WEBHOOK_ID,
                "url": "https://example.com/webhooks/exa",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("webhook_id", SAMPLE_WEBHOOK_ID),
                ("webhook", _SAMPLE_MODEL),
            ],
            test_mock={"call_exa": lambda *args, **kwargs: SAMPLE_WEBHOOK},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.call_exa(
            credentials,
            "PATCH",
            f"{WEBHOOKS_URL}/{path_segment(input_data.webhook_id)}",
            body=build_update_body(input_data),
        )
        webhook = WebsetWebhookModel.from_api(data)
        yield "webhook_id", webhook.id
        yield "webhook", webhook


def build_update_body(input_data: ExaUpdateWebsetWebhookBlock.Input) -> dict[str, Any]:
    """Build the webhooks-update body; omitted fields stay unchanged at Exa."""
    body: dict[str, Any] = {}
    if input_data.url and input_data.url.strip():
        body["url"] = input_data.url.strip()
    if input_data.events:
        body["events"] = event_type_values(input_data.events)
    if input_data.metadata is not None:
        body["metadata"] = input_data.metadata
    if not body:
        raise ValueError("Nothing to update: set a URL, events or metadata")
    return body


class ExaDeleteWebsetWebhookBlock(WebsetsTeamApiMixin, Block):
    """Delete a Websets webhook, stopping its deliveries."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webhook_id: str = SchemaField(
            description="The webhook to delete. This cannot be undone.",
            placeholder=SAMPLE_WEBHOOK_ID,
        )

    class Output(BlockSchemaOutput):
        webhook_id: str = SchemaField(description="ID of the deleted webhook")
        status: str = SchemaField(description="The webhook's status, now inactive")

    def __init__(self):
        super().__init__(
            id="d118828a-ce11-4f38-9014-ecf6a2654fb5",
            description="Delete an Exa Websets webhook so Exa stops delivering to it",
            categories={BlockCategory.SEARCH},
            input_schema=ExaDeleteWebsetWebhookBlock.Input,
            output_schema=ExaDeleteWebsetWebhookBlock.Output,
            is_irreversible_action=True,
            effect=BlockEffect.EXTERNAL,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "webhook_id": SAMPLE_WEBHOOK_ID,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("webhook_id", SAMPLE_WEBHOOK_ID), ("status", "inactive")],
            test_mock={
                "call_exa": lambda *args, **kwargs: {
                    **SAMPLE_WEBHOOK,
                    "status": "inactive",
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.call_exa(
            credentials,
            "DELETE",
            f"{WEBHOOKS_URL}/{path_segment(input_data.webhook_id)}",
        )
        yield "webhook_id", data.get("id") or input_data.webhook_id
        yield "status", data.get("status", "inactive")
