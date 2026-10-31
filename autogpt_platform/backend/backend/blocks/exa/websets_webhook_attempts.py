"""
Exa Websets webhook listing blocks: list webhooks and list delivery attempts.
"""

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
from ._exa_rest import QueryParams, add_param, path_segment
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._webhook import ExaEventType
from ._websets_webhook_samples import SAMPLE_ATTEMPT, SAMPLE_WEBHOOK, SAMPLE_WEBHOOK_ID
from .websets_webhooks_models import (
    WEBHOOKS_URL,
    WebhookAttemptModel,
    WebsetsTeamApiMixin,
    WebsetWebhookModel,
)

_SAMPLE_WEBHOOK_MODEL = WebsetWebhookModel.from_api(SAMPLE_WEBHOOK)
_SAMPLE_ATTEMPT_MODEL = WebhookAttemptModel.from_api(SAMPLE_ATTEMPT)


class ExaListWebsetWebhooksBlock(WebsetsTeamApiMixin, Block):
    """List the team's active Websets webhooks."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        limit: int = SchemaField(
            default=25, description="Webhooks per page (1-200)", ge=1, le=200
        )
        cursor: str = SchemaField(
            default="", description="Cursor from a previous page's next_cursor"
        )

    class Output(BlockSchemaOutput):
        webhooks: list[WebsetWebhookModel] = SchemaField(
            description="Webhooks on this page"
        )
        webhook: WebsetWebhookModel = SchemaField(
            title="Webhook", description="Each webhook, one at a time"
        )
        has_more: bool = SchemaField(description="Whether another page exists")
        next_cursor: str | None = SchemaField(
            description="Pass as cursor to fetch the next page"
        )

    def __init__(self):
        super().__init__(
            id="6b45f0e9-78d2-49ce-9de9-da655b070a03",
            description="List your team's active Exa Websets webhooks",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListWebsetWebhooksBlock.Input,
            output_schema=ExaListWebsetWebhooksBlock.Output,
            effect=BlockEffect.READ,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("webhooks", [_SAMPLE_WEBHOOK_MODEL]),
                ("webhook", _SAMPLE_WEBHOOK_MODEL),
                ("has_more", False),
                ("next_cursor", None),
            ],
            test_mock={
                "call_exa": lambda *args, **kwargs: {
                    "data": [SAMPLE_WEBHOOK],
                    "hasMore": False,
                    "nextCursor": None,
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        params: QueryParams = []
        add_param(params, "limit", input_data.limit)
        add_param(params, "cursor", input_data.cursor)
        data = await self.call_exa(credentials, "GET", WEBHOOKS_URL, params=params)
        webhooks = [WebsetWebhookModel.from_api(item) for item in data.get("data", [])]
        yield "webhooks", webhooks
        for webhook in webhooks:
            yield "webhook", webhook
        yield "has_more", data.get("hasMore", False)
        yield "next_cursor", data.get("nextCursor")


class ExaListWebsetWebhookAttemptsBlock(WebsetsTeamApiMixin, Block):
    """List a Websets webhook's delivery attempts, newest first."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webhook_id: str = SchemaField(
            description="The webhook whose deliveries to list",
            placeholder=SAMPLE_WEBHOOK_ID,
        )
        event_type: ExaEventType | None = SchemaField(
            default=None,
            description="Only list attempts for this event type",
            advanced=False,
        )
        successful: bool | None = SchemaField(
            default=None,
            description="Only list successful (true) or failed (false) attempts; "
            "leave empty for both",
            advanced=False,
        )
        limit: int = SchemaField(
            default=25, description="Attempts per page (1-200)", ge=1, le=200
        )
        cursor: str = SchemaField(
            default="", description="Cursor from a previous page's next_cursor"
        )

    class Output(BlockSchemaOutput):
        attempts: list[WebhookAttemptModel] = SchemaField(
            description="Delivery attempts on this page, newest first"
        )
        attempt: WebhookAttemptModel = SchemaField(
            title="Attempt", description="Each attempt, one at a time"
        )
        has_more: bool = SchemaField(description="Whether another page exists")
        next_cursor: str | None = SchemaField(
            description="Pass as cursor to fetch the next page"
        )

    def __init__(self):
        super().__init__(
            id="7597e071-1b43-4973-825d-859796660e1d",
            description="List an Exa Websets webhook's delivery attempts, with "
            "status codes and responses, to debug failed deliveries",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListWebsetWebhookAttemptsBlock.Input,
            output_schema=ExaListWebsetWebhookAttemptsBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "webhook_id": SAMPLE_WEBHOOK_ID,
                "successful": False,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("attempts", [_SAMPLE_ATTEMPT_MODEL]),
                ("attempt", _SAMPLE_ATTEMPT_MODEL),
                ("has_more", False),
                ("next_cursor", None),
            ],
            test_mock={
                "call_exa": lambda *args, **kwargs: {
                    "data": [SAMPLE_ATTEMPT],
                    "hasMore": False,
                    "nextCursor": None,
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.call_exa(
            credentials,
            "GET",
            f"{WEBHOOKS_URL}/{path_segment(input_data.webhook_id)}/attempts",
            params=attempt_params(input_data),
        )
        attempts = [WebhookAttemptModel.from_api(item) for item in data.get("data", [])]
        yield "attempts", attempts
        for attempt in attempts:
            yield "attempt", attempt
        yield "has_more", data.get("hasMore", False)
        yield "next_cursor", data.get("nextCursor")


def attempt_params(input_data: ExaListWebsetWebhookAttemptsBlock.Input) -> QueryParams:
    params: QueryParams = []
    add_param(
        params,
        "eventType",
        input_data.event_type.value if input_data.event_type else None,
    )
    add_param(params, "successful", input_data.successful)
    add_param(params, "limit", input_data.limit)
    add_param(params, "cursor", input_data.cursor)
    return params
