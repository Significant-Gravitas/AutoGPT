"""
Exa Websets event blocks: list events and get one event.

Exa records an event for every change in your Websets (and Websets monitors)
and keeps it for 60 days. Listing them is a polling alternative to the
"Exa Webset Webhook" trigger block.
"""

from datetime import UTC, datetime

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
from ._websets_webhook_samples import SAMPLE_EVENT, SAMPLE_EVENT_ID
from .websets_webhooks_models import (
    EVENTS_URL,
    WebsetEventModel,
    WebsetsTeamApiMixin,
    event_type_values,
)

_SAMPLE_EVENT_MODEL = WebsetEventModel.from_api(SAMPLE_EVENT)


class ExaListWebsetEventsBlock(WebsetsTeamApiMixin, Block):
    """List Websets events, newest first, with type and date filters."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        types: list[ExaEventType] = SchemaField(
            default_factory=list,
            description="Only list these event types; leave empty for all",
            advanced=False,
        )
        created_after: datetime | None = SchemaField(
            default=None,
            description="Only events at or after this time (UTC if no timezone)",
            advanced=False,
        )
        created_before: datetime | None = SchemaField(
            default=None,
            description="Only events at or before this time (UTC if no timezone)",
        )
        limit: int = SchemaField(
            default=25, description="Events per page (1-200)", ge=1, le=200
        )
        cursor: str = SchemaField(
            default="", description="Cursor from a previous page's next_cursor"
        )

    class Output(BlockSchemaOutput):
        events: list[WebsetEventModel] = SchemaField(description="Events on this page")
        event: WebsetEventModel = SchemaField(
            title="Event", description="Each event, one at a time"
        )
        has_more: bool = SchemaField(description="Whether another page exists")
        next_cursor: str | None = SchemaField(
            description="Pass as cursor to fetch the next page"
        )

    def __init__(self):
        super().__init__(
            id="395d074c-3ddd-4f85-81a6-0344c280032a",
            description="List Exa Websets events (new items, finished searches, "
            "monitor runs and more) from the last 60 days",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListWebsetEventsBlock.Input,
            output_schema=ExaListWebsetEventsBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "types": [ExaEventType.WEBSET_ITEM_CREATED],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("events", [_SAMPLE_EVENT_MODEL]),
                ("event", _SAMPLE_EVENT_MODEL),
                ("has_more", False),
                ("next_cursor", None),
            ],
            test_mock={
                "call_exa": lambda *args, **kwargs: {
                    "data": [SAMPLE_EVENT],
                    "hasMore": False,
                    "nextCursor": None,
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.call_exa(
            credentials, "GET", EVENTS_URL, params=event_params(input_data)
        )
        events = [WebsetEventModel.from_api(item) for item in data.get("data", [])]
        yield "events", events
        for event in events:
            yield "event", event
        yield "has_more", data.get("hasMore", False)
        yield "next_cursor", data.get("nextCursor")


def event_params(input_data: ExaListWebsetEventsBlock.Input) -> QueryParams:
    """Query string for events-list; ``types`` repeats once per type."""
    params: QueryParams = [
        ("types", value) for value in event_type_values(input_data.types)
    ]
    add_param(params, "createdAfter", iso_utc(input_data.created_after))
    add_param(params, "createdBefore", iso_utc(input_data.created_before))
    add_param(params, "limit", input_data.limit)
    add_param(params, "cursor", input_data.cursor)
    return params


def iso_utc(value: datetime | None) -> str | None:
    """Format a datetime as ISO 8601 UTC, treating naive values as UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


class ExaGetWebsetEventBlock(WebsetsTeamApiMixin, Block):
    """Fetch one Websets event by ID."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        event_id: str = SchemaField(
            description="The event to fetch", placeholder=SAMPLE_EVENT_ID
        )

    class Output(BlockSchemaOutput):
        event_id: str = SchemaField(description="The event's ID")
        event_type: str = SchemaField(description="e.g. webset.item.created")
        created_at: str = SchemaField(description="When it happened (ISO 8601)")
        data: dict = SchemaField(
            description="The full resource the event is about (webset, item, "
            "search, import or monitor)"
        )
        event: WebsetEventModel = SchemaField(
            title="Event", description="The full event"
        )

    def __init__(self):
        super().__init__(
            id="c10a2234-59b2-46d0-a6af-b6bef3b17455",
            description="Get one Exa Websets event and the resource it is about",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetWebsetEventBlock.Input,
            output_schema=ExaGetWebsetEventBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "event_id": SAMPLE_EVENT_ID,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("event_id", SAMPLE_EVENT_ID),
                ("event_type", "webset.item.created"),
                ("created_at", SAMPLE_EVENT["createdAt"]),
                ("data", SAMPLE_EVENT["data"]),
                ("event", _SAMPLE_EVENT_MODEL),
            ],
            test_mock={"call_exa": lambda *args, **kwargs: SAMPLE_EVENT},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self.call_exa(
            credentials, "GET", f"{EVENTS_URL}/{path_segment(input_data.event_id)}"
        )
        event = WebsetEventModel.from_api(data)
        yield "event_id", event.id
        yield "event_type", event.type
        yield "created_at", event.created_at
        yield "data", event.data
        yield "event", event
