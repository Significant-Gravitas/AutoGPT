import time
from datetime import datetime
from enum import Enum
from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import exa
from ._exa_rest import ExaApiError
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .websets_api import ExaWebsetsBlockBase, segment
from .websets_fixtures import WEBSET_ID, sample_preview, sample_webset
from .websets_models import (
    EnrichmentSummaryModel,
    MonitorSummaryModel,
    PreviewCriterionModel,
    PreviewEnrichmentModel,
    PreviewWebsetModel,
    SearchSummaryModel,
    Webset,
    WebsetStatisticsModel,
)
from .websets_payloads import (
    criteria_payload,
    enrichment_payload,
    entity_payload,
    import_payload,
    scope_payload,
    sources_payload,
)
from .websets_types import ApiWebset

__all__ = [
    "PreviewCriterionModel",
    "PreviewEnrichmentModel",
    "PreviewWebsetModel",
    "Webset",
]


class SearchEntityType(str, Enum):
    COMPANY = "company"
    PERSON = "person"
    ARTICLE = "article"
    RESEARCH_PAPER = "research_paper"
    CUSTOM = "custom"
    AUTO = "auto"


class SearchType(str, Enum):
    IMPORT = "import"
    WEBSET = "webset"


class EnrichmentFormat(str, Enum):
    TEXT = "text"
    DATE = "date"
    NUMBER = "number"
    OPTIONS = "options"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"


class ExaCreateWebsetBlock(ExaWebsetsBlockBase):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )

        # Search parameters (flattened)
        search_query: str = SchemaField(
            description="What to find, in plain language (up to 5,000 characters). Be specific about the characteristics every result must have. Any URL provided will be crawled and used as context for the search.",
            placeholder="Marketing agencies based in the US, that focus on consumer products",
        )
        title: str | None = SchemaField(
            default=None,
            description="Name shown wherever the webset is displayed. Leave empty to let Exa generate one.",
            placeholder="US consumer marketing agencies",
            advanced=False,
        )
        search_count: int | None = SchemaField(
            default=10,
            description="Number of items the search will attempt to find. The actual number of items found may be less than this number depending on the search complexity.",
            ge=1,
            le=1000,
        )
        search_entity_type: SearchEntityType = SchemaField(
            default=SearchEntityType.AUTO,
            description="Kind of result to find: company, person, article, research_paper, or custom (describe it below). 'auto' lets Exa detect it from the query.",
            advanced=True,
        )
        search_entity_description: str | None = SchemaField(
            default=None,
            description="What each result is, when the entity type is 'custom' (2-200 characters, required for custom).",
            placeholder="Open-source LLM repositories",
            advanced=True,
        )

        # Search criteria (flattened)
        search_criteria: list[str] = SchemaField(
            default_factory=list,
            description="Up to 5 criteria every item must satisfy (each up to 1,000 characters). Leave empty to have Exa generate them from the query.",
            advanced=True,
        )
        search_recall: bool = SchemaField(
            default=False,
            description="Also estimate how many matching results exist on the web. The estimate appears in the search's 'recall' field.",
            advanced=True,
        )
        search_max_people_per_company: int | None = SchemaField(
            default=None,
            description="For people searches: try to include at most this many people from the same current employer (soft cap).",
            ge=1,
            advanced=True,
        )

        # Search exclude sources (flattened)
        search_exclude_sources: list[str] = SchemaField(
            default_factory=list,
            description="IDs of imports or websets whose items this search must leave out",
            advanced=True,
        )
        search_exclude_types: list[SearchType] = SchemaField(
            default_factory=list,
            description="Source type for each ID above, in the same order ('import' or 'webset'; defaults to 'import')",
            advanced=True,
        )

        # Search scope sources (flattened)
        search_scope_sources: list[str] = SchemaField(
            default_factory=list,
            description="IDs of imports or websets to search within instead of the whole web",
            advanced=True,
        )
        search_scope_types: list[SearchType] = SchemaField(
            default_factory=list,
            description="Source type for each scope ID, in the same order ('import' or 'webset'; defaults to 'import')",
            advanced=True,
        )
        search_scope_relationships: list[str] = SchemaField(
            default_factory=list,
            description="Optional, one per scope source: how the results relate to that source's items, for hop searches (e.g. 'investors of these companies')",
            advanced=True,
        )
        search_scope_relationship_limits: list[int] = SchemaField(
            default_factory=list,
            description="Required for each relationship above: how many related entities to find per source item (1-10)",
            advanced=True,
        )

        # Import parameters (flattened)
        import_sources: list[str] = SchemaField(
            default_factory=list,
            description="IDs of imports or websets whose items are copied into this webset",
            advanced=True,
        )
        import_types: list[SearchType] = SchemaField(
            default_factory=list,
            description="Source type for each import ID, in the same order ('import' or 'webset'; defaults to 'import')",
            advanced=True,
        )
        import_evaluate: list[bool] = SchemaField(
            default_factory=list,
            description="Optional, one per import source: check that source's items against the search criteria (requires a search)",
            advanced=True,
        )

        # Webset-wide exclusions
        exclude_sources: list[str] = SchemaField(
            default_factory=list,
            description="IDs of imports or websets whose items are left out of every search and import in this webset",
            advanced=True,
        )
        exclude_types: list[SearchType] = SchemaField(
            default_factory=list,
            description="Source type for each webset-wide exclude ID, in the same order ('import' or 'webset'; defaults to 'import')",
            advanced=True,
        )

        # Enrichment parameters (flattened)
        enrichment_descriptions: list[str] = SchemaField(
            default_factory=list,
            description="Data to extract for every item, one description per enrichment column (up to 5,000 characters each)",
            advanced=True,
        )
        enrichment_formats: list[EnrichmentFormat] = SchemaField(
            default_factory=list,
            description="Format for each enrichment, in the same order: text, date, number, options, email, phone or url. If not specified, Exa picks the best format.",
            advanced=True,
        )
        enrichment_options: list[list[str]] = SchemaField(
            default_factory=list,
            description="For 'options' enrichments, in the same order: the labels to choose from (1-150 per enrichment)",
            advanced=True,
        )
        enrichment_metadata: list[dict] = SchemaField(
            default_factory=list,
            description="Optional metadata for each enrichment, in the same order",
            advanced=True,
        )

        # Webset metadata
        external_id: str | None = SchemaField(
            default=None,
            description="Your own identifier for the webset (up to 300 characters). Other Websets blocks accept it in place of the webset ID.",
            placeholder="my-webset-123",
            advanced=True,
        )
        metadata: dict | None = SchemaField(
            default_factory=dict,
            description="Key-value pairs to store on the webset (values up to 1,000 characters)",
            advanced=True,
        )

        # Polling parameters
        wait_for_initial_results: bool = SchemaField(
            default=True,
            description="Wait for the initial search to complete before returning. This ensures you get results immediately.",
        )
        polling_timeout: int = SchemaField(
            default=300,
            description="Maximum time to wait for completion in seconds (only used if wait_for_initial_results is True)",
            advanced=True,
            ge=1,
            le=600,
        )

    class Output(BlockSchemaOutput):
        webset: Webset = SchemaField(description="The created webset with full details")
        initial_item_count: int | None = SchemaField(
            description="Number of items found in the initial search (only if wait_for_initial_results was True)"
        )
        completion_time: float | None = SchemaField(
            description="Time taken to complete the initial search in seconds (only if wait_for_initial_results was True)"
        )

    def __init__(self):
        super().__init__(
            id="0cda29ff-c549-4a19-8805-c982b7d4ec34",
            description="Create a new Exa Webset for persistent web search collections with optional waiting for initial results",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCreateWebsetBlock.Input,
            output_schema=ExaCreateWebsetBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "search_query": "AI startups in Europe",
                "title": "AI startups",
                "wait_for_initial_results": False,
            },
            test_output=[("webset", Webset.model_validate(sample_webset()))],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_api": lambda *args, **kwargs: sample_webset()},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        payload = create_webset_payload(input_data)
        start_time = time.time()
        created = await self._api(credentials, "POST", "/websets", json=payload)
        webset = Webset.model_validate(created)

        if not (input_data.wait_for_initial_results and "search" in payload):
            yield "webset", webset
            return

        final_webset = await self._wait_until_idle(
            credentials, webset.id, input_data.polling_timeout
        )
        yield "webset", webset
        yield "initial_item_count", final_webset.items_found
        yield "completion_time", time.time() - start_time


def create_webset_payload(data: ExaCreateWebsetBlock.Input) -> dict[str, Any]:
    """Build the ``POST /v0/websets`` body from the block's flat inputs."""
    payload: dict[str, Any] = {}
    if data.search_query:
        payload["search"] = _search_payload(data)
    imports = import_payload(
        data.import_sources, [t.value for t in data.import_types], data.import_evaluate
    )
    if imports:
        payload["import"] = imports
    enrichments = _enrichments_payload(data)
    if enrichments:
        payload["enrichments"] = enrichments
    exclude = sources_payload(
        data.exclude_sources, [t.value for t in data.exclude_types]
    )
    if exclude:
        payload["exclude"] = exclude
    if data.title:
        payload["title"] = data.title
    if data.external_id:
        payload["externalId"] = data.external_id
    if data.metadata:
        payload["metadata"] = data.metadata
    return payload


def _search_payload(data: ExaCreateWebsetBlock.Input) -> dict[str, Any]:
    search: dict[str, Any] = {"query": data.search_query}
    optional = {
        "count": data.search_count,
        "entity": entity_payload(
            data.search_entity_type.value, data.search_entity_description
        ),
        "criteria": criteria_payload(data.search_criteria),
        "exclude": sources_payload(
            data.search_exclude_sources, [t.value for t in data.search_exclude_types]
        ),
        "scope": scope_payload(
            data.search_scope_sources,
            [t.value for t in data.search_scope_types],
            data.search_scope_relationships,
            data.search_scope_relationship_limits,
        ),
        "maxPeoplePerCompany": data.search_max_people_per_company,
        "recall": True if data.search_recall else None,
    }
    search.update({k: v for k, v in optional.items() if v})
    return search


def _enrichments_payload(data: ExaCreateWebsetBlock.Input) -> list[dict[str, Any]]:
    def at(values: list, idx: int, default: Any) -> Any:
        return values[idx] if idx < len(values) else default

    return [
        enrichment_payload(
            description,
            fmt.value if (fmt := at(data.enrichment_formats, idx, None)) else None,
            at(data.enrichment_options, idx, []),
            at(data.enrichment_metadata, idx, None),
        )
        for idx, description in enumerate(data.enrichment_descriptions)
    ]


class ExaCreateOrFindWebsetBlock(ExaWebsetsBlockBase):
    """Create a new webset or return existing one if external_id already exists (idempotent)."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )

        external_id: str = SchemaField(
            description="External identifier for this webset - used to find existing or create new",
            placeholder="my-unique-webset-id",
        )

        search_query: str | None = SchemaField(
            default=None,
            description="Search query (optional - only needed if creating new webset)",
            placeholder="Marketing agencies based in the US",
        )
        search_count: int = SchemaField(
            default=10,
            description="Number of items to find in initial search",
            ge=1,
            le=1000,
        )

        metadata: dict | None = SchemaField(
            default=None,
            description="Key-value pairs to associate with the webset",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        webset: Webset = SchemaField(
            description="The webset (existing or newly created)"
        )
        was_created: bool = SchemaField(
            description="True if webset was newly created, False if it already existed"
        )

    def __init__(self):
        super().__init__(
            id="214542b6-3603-4bea-bc07-f51c2871cbd9",
            description="Create a new webset or return existing one by external_id (idempotent operation)",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCreateOrFindWebsetBlock.Input,
            output_schema=ExaCreateOrFindWebsetBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            existing = await self._api(
                credentials, "GET", f"/websets/{segment(input_data.external_id)}"
            )
        except ExaApiError as e:
            if e.status != 404:
                raise
        else:
            yield "webset", Webset.model_validate(existing)
            yield "was_created", False
            return

        payload: dict[str, Any] = {"externalId": input_data.external_id}
        if input_data.search_query:
            payload["search"] = {
                "query": input_data.search_query,
                "count": input_data.search_count,
            }
        if input_data.metadata:
            payload["metadata"] = input_data.metadata
        created = await self._api(credentials, "POST", "/websets", json=payload)
        yield "webset", Webset.model_validate(created)
        yield "was_created", True


class ExaUpdateWebsetBlock(ExaWebsetsBlockBase):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset to update",
            placeholder="webset-id-or-external-id",
        )
        title: str | None = SchemaField(
            default=None,
            description="New name for the webset. Leave empty to keep the current one.",
            placeholder="US consumer marketing agencies",
            advanced=False,
        )
        metadata: dict | None = SchemaField(
            default=None,
            description="Key-value pairs to associate with this webset (set to null to clear)",
        )

    class Output(BlockSchemaOutput):
        webset_id: str = SchemaField(description="The unique identifier for the webset")
        status: str = SchemaField(description="The status of the webset")
        external_id: str | None = SchemaField(
            description="The external identifier for the webset"
        )
        title: str | None = SchemaField(description="The webset's name")
        metadata: dict = SchemaField(description="Updated metadata for the webset")
        updated_at: str = SchemaField(
            description="The date and time the webset was updated"
        )

    def __init__(self):
        super().__init__(
            id="89ccd99a-3c2b-4fbf-9e25-0ffa398d0314",
            description="Update the title or metadata of an existing Webset",
            categories={BlockCategory.SEARCH},
            input_schema=ExaUpdateWebsetBlock.Input,
            output_schema=ExaUpdateWebsetBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "webset_id": WEBSET_ID,
                "title": "AI startups",
            },
            test_output=[
                ("webset_id", WEBSET_ID),
                ("status", "idle"),
                ("external_id", None),
                ("title", "AI startups"),
                ("metadata", {}),
                ("updated_at", "2024-01-01T00:00:00Z"),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_api": lambda *args, **kwargs: sample_webset()},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        payload: dict[str, Any] = {}
        if input_data.title:
            payload["title"] = input_data.title
        if input_data.metadata is not None:
            payload["metadata"] = input_data.metadata

        data = await self._api(
            credentials,
            "POST",
            f"/websets/{segment(input_data.webset_id)}",
            json=payload,
        )
        webset = ApiWebset.model_validate(data)

        yield "webset_id", webset.id
        yield "status", webset.status
        yield "external_id", webset.external_id
        yield "title", webset.title
        yield "metadata", webset.metadata or {}
        yield "updated_at", webset.updated_at or ""


class ExaListWebsetsBlock(ExaWebsetsBlockBase):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        trigger: Any | None = SchemaField(
            default=None,
            description="Trigger for the webset, value is ignored!",
            advanced=False,
        )
        search: str | None = SchemaField(
            default=None,
            description="Only return websets whose ID, external ID or title matches this text (2-50 characters)",
            placeholder="startups",
            min_length=2,
            max_length=50,
            advanced=False,
        )
        cursor: str | None = SchemaField(
            default=None,
            description="Cursor for pagination through results",
            advanced=True,
        )
        limit: int = SchemaField(
            default=25,
            description="Number of websets to return (1-100)",
            ge=1,
            le=100,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        websets: list[Webset] = SchemaField(description="List of websets")
        webset: Webset = SchemaField(
            description="Individual webset (yielded for each webset in the list)"
        )
        has_more: bool = SchemaField(
            description="Whether there are more results to paginate through"
        )
        next_cursor: str | None = SchemaField(
            description="Cursor for the next page of results"
        )

    def __init__(self):
        super().__init__(
            id="1dcd8fd6-c13f-4e6f-bd4c-654428fa4757",
            description="List all Websets with pagination support, optionally filtered by ID, external ID or title",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListWebsetsBlock.Input,
            output_schema=ExaListWebsetsBlock.Output,
            effect=BlockEffect.READ,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "search": "AI"},
            test_output=[
                ("websets", [Webset.model_validate(sample_webset())]),
                ("webset", Webset.model_validate(sample_webset())),
                ("has_more", False),
                ("next_cursor", None),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "_api": lambda *args, **kwargs: {
                    "data": [sample_webset()],
                    "hasMore": False,
                    "nextCursor": None,
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        params = {
            "cursor": input_data.cursor,
            "limit": input_data.limit,
            "search": input_data.search or None,
        }
        response = await self._api(credentials, "GET", "/websets", params=params)
        websets = [Webset.model_validate(w) for w in response.get("data", [])]

        yield "websets", websets
        for webset in websets:
            yield "webset", webset
        yield "has_more", bool(response.get("hasMore"))
        yield "next_cursor", response.get("nextCursor")


class ExaGetWebsetBlock(ExaWebsetsBlockBase):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset to retrieve",
            placeholder="webset-id-or-external-id",
        )
        expand_items: bool = SchemaField(
            default=False,
            description="Also return the webset's first 100 items in the 'items' output",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        webset_id: str = SchemaField(description="The unique identifier for the webset")
        status: str = SchemaField(
            description="The status of the webset: idle, pending, running or paused"
        )
        external_id: str | None = SchemaField(
            description="The external identifier for the webset"
        )
        title: str | None = SchemaField(description="The webset's name")
        dashboard_url: str | None = SchemaField(
            description="Link to the webset in the Exa dashboard"
        )
        searches: list[dict] = SchemaField(
            description="The searches performed on the webset"
        )
        imports: list[dict] = SchemaField(
            description="The imports that brought items into the webset"
        )
        enrichments: list[dict] = SchemaField(
            description="The enrichments applied to the webset"
        )
        monitors: list[dict] = SchemaField(description="The monitors for the webset")
        excludes: list[dict] = SchemaField(
            description="Imports or websets excluded from every operation in this webset"
        )
        items: list[dict] = SchemaField(
            description="Up to the first 100 items (empty unless 'expand items' is on)"
        )
        metadata: dict = SchemaField(
            description="Key-value pairs associated with the webset"
        )
        created_at: str = SchemaField(
            description="The date and time the webset was created"
        )
        updated_at: str = SchemaField(
            description="The date and time the webset was last updated"
        )

    def __init__(self):
        super().__init__(
            id="6ab8e12a-132c-41bf-b5f3-d662620fa832",
            description="Retrieve a Webset by ID or external ID",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetWebsetBlock.Input,
            output_schema=ExaGetWebsetBlock.Output,
            effect=BlockEffect.READ,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "webset_id": WEBSET_ID},
            test_output=[
                ("webset_id", WEBSET_ID),
                ("status", "idle"),
                ("external_id", None),
                ("title", "AI startups"),
                ("dashboard_url", f"https://websets.exa.ai/{WEBSET_ID}"),
                ("searches", list),
                ("imports", []),
                ("enrichments", []),
                ("monitors", []),
                ("excludes", []),
                ("items", []),
                ("metadata", {}),
                ("created_at", "2024-01-01T00:00:00Z"),
                ("updated_at", "2024-01-01T00:00:00Z"),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_api": lambda *args, **kwargs: sample_webset()},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self._api(
            credentials,
            "GET",
            f"/websets/{segment(input_data.webset_id)}",
            params={"expand": "items" if input_data.expand_items else None},
        )
        webset = ApiWebset.model_validate(data)

        yield "webset_id", webset.id
        yield "status", webset.status
        yield "external_id", webset.external_id
        yield "title", webset.title
        yield "dashboard_url", webset.dashboard_url
        for key in ("searches", "imports", "enrichments", "monitors", "excludes"):
            yield key, data.get(key) or []
        yield "items", data.get("items") or []
        yield "metadata", webset.metadata or {}
        yield "created_at", webset.created_at or ""
        yield "updated_at", webset.updated_at or ""


class ExaDeleteWebsetBlock(ExaWebsetsBlockBase):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset to delete",
            placeholder="webset-id-or-external-id",
        )

    class Output(BlockSchemaOutput):
        webset_id: str = SchemaField(
            description="The unique identifier for the deleted webset"
        )
        external_id: str | None = SchemaField(
            description="The external identifier for the deleted webset"
        )
        status: str = SchemaField(description="The status of the deleted webset")
        success: str = SchemaField(description="Whether the deletion was successful")

    def __init__(self):
        super().__init__(
            id="aa6994a2-e986-421f-8d4c-7671d3be7b7e",
            description="Delete a Webset and all its items",
            categories={BlockCategory.SEARCH},
            input_schema=ExaDeleteWebsetBlock.Input,
            output_schema=ExaDeleteWebsetBlock.Output,
            is_irreversible_action=True,
            effect=BlockEffect.EXTERNAL,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self._api(
            credentials, "DELETE", f"/websets/{segment(input_data.webset_id)}"
        )
        webset = ApiWebset.model_validate(data)

        yield "webset_id", webset.id
        yield "external_id", webset.external_id
        yield "status", webset.status
        yield "success", "true"


class ExaCancelWebsetBlock(ExaWebsetsBlockBase):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset to cancel",
            placeholder="webset-id-or-external-id",
        )

    class Output(BlockSchemaOutput):
        webset_id: str = SchemaField(description="The unique identifier for the webset")
        status: str = SchemaField(
            description="The status of the webset after cancellation"
        )
        external_id: str | None = SchemaField(
            description="The external identifier for the webset"
        )
        success: str = SchemaField(
            description="Whether the cancellation was successful"
        )

    def __init__(self):
        super().__init__(
            id="e40a6420-1db8-47bb-b00a-0e6aecd74176",
            description="Cancel all operations being performed on a Webset",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCancelWebsetBlock.Input,
            output_schema=ExaCancelWebsetBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = await self._api(
            credentials, "POST", f"/websets/{segment(input_data.webset_id)}/cancel"
        )
        webset = ApiWebset.model_validate(data)

        yield "webset_id", webset.id
        yield "status", webset.status
        yield "external_id", webset.external_id
        yield "success", "true"


class ExaPreviewWebsetBlock(ExaWebsetsBlockBase):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        query: str = SchemaField(
            description="Your search query to preview. Use this to see how Exa will interpret your search before creating a webset.",
            placeholder="Marketing agencies based in the US, with brands worked with and city",
        )
        entity_type: SearchEntityType | None = SchemaField(
            default=None,
            description="Entity type to force: 'company', 'person', 'article', 'research_paper', or 'custom'. If not provided, Exa will auto-detect.",
            advanced=True,
        )
        entity_description: str | None = SchemaField(
            default=None,
            description="Description for custom entity type (2-200 characters, required when entity_type is 'custom')",
            advanced=True,
        )
        include_sample_items: bool = SchemaField(
            default=False,
            description="Also search for a few matching items and return them in 'items'. This runs a real search and uses Websets credits.",
            advanced=True,
        )
        sample_count: int = SchemaField(
            default=10,
            description="How many sample items to return when 'include sample items' is on (1-10)",
            ge=1,
            le=10,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        preview: PreviewWebsetModel = SchemaField(
            description="Full preview response with search and enrichment details"
        )
        entity_type: str = SchemaField(
            description="The detected or specified entity type"
        )
        entity_description: str | None = SchemaField(
            description="Description of the entity type"
        )
        criteria: list[PreviewCriterionModel] = SchemaField(
            description="Generated search criteria that will be used"
        )
        enrichment_columns: list[PreviewEnrichmentModel] = SchemaField(
            description="Available enrichment columns that can be extracted"
        )
        items: list[dict] = SchemaField(
            description="Sample matching items (empty unless 'include sample items' is on)"
        )
        interpretation: str = SchemaField(
            description="Human-readable interpretation of how the query will be processed"
        )
        suggestions: list[str] = SchemaField(
            description="Suggestions for improving the query"
        )

    def __init__(self):
        super().__init__(
            id="f8c4e2a1-9b3d-4e5f-a6c7-d8e9f0a1b2c3",
            description="Preview how a search query will be interpreted before creating a webset. Helps understand entity detection, criteria generation, and available enrichments.",
            categories={BlockCategory.SEARCH},
            input_schema=ExaPreviewWebsetBlock.Input,
            output_schema=ExaPreviewWebsetBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "AI startups in Europe",
            },
            test_output=[
                ("preview", PreviewWebsetModel.from_api(sample_preview())),
                ("entity_type", "company"),
                ("entity_description", None),
                ("criteria", [PreviewCriterionModel(description="Based in Europe")]),
                ("enrichment_columns", list),
                ("items", []),
                (
                    "interpretation",
                    (
                        "Query will search for company with 1 criteria and 1 "
                        "available enrichment columns"
                    ),
                ),
                ("suggestions", []),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_api": lambda *args, **kwargs: sample_preview()},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        search: dict[str, Any] = {"query": input_data.query}
        entity_type = input_data.entity_type.value if input_data.entity_type else None
        entity = entity_payload(entity_type, input_data.entity_description)
        if entity:
            search["entity"] = entity
        if input_data.include_sample_items:
            search["count"] = input_data.sample_count

        data = await self._api(
            credentials,
            "POST",
            "/websets/preview",
            json={"search": search},
            params={"search": True if input_data.include_sample_items else None},
        )
        preview = PreviewWebsetModel.from_api(data)
        criteria = preview.search.criteria
        enrichments = preview.enrichments

        yield "preview", preview
        yield "entity_type", preview.search.entity_type
        yield "entity_description", preview.search.entity_description
        yield "criteria", criteria
        yield "enrichment_columns", enrichments
        yield "items", preview.items
        yield (
            "interpretation",
            _interpretation(
                preview.search.entity_type,
                preview.search.entity_description,
                len(criteria),
                len(enrichments),
            ),
        )
        yield "suggestions", _suggestions(len(criteria), len(enrichments))


def _interpretation(
    entity_type: str, entity_description: str | None, criteria: int, columns: int
) -> str:
    text = f"Query will search for {entity_type}"
    if entity_description:
        text += f" ({entity_description})"
    if criteria:
        text += f" with {criteria} criteria"
    if columns:
        text += f" and {columns} available enrichment columns"
    return text


def _suggestions(criteria: int, columns: int) -> list[str]:
    suggestions = []
    if not criteria:
        suggestions.append("Consider adding specific criteria to narrow your search")
    if not columns:
        suggestions.append("Consider specifying what data points you want to extract")
    return suggestions


class ExaWebsetStatusBlock(ExaWebsetsBlockBase):
    """Get a quick status overview of a webset without fetching all details."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )

    class Output(BlockSchemaOutput):
        webset_id: str = SchemaField(description="The webset identifier")
        status: str = SchemaField(
            description="Current status (idle, pending, running or paused)"
        )
        item_count: int = SchemaField(description="Total number of items in the webset")
        search_count: int = SchemaField(description="Number of searches performed")
        enrichment_count: int = SchemaField(
            description="Number of enrichments configured"
        )
        monitor_count: int = SchemaField(description="Number of monitors configured")
        last_updated: str = SchemaField(description="When the webset was last updated")
        is_processing: bool = SchemaField(
            description="Whether any operations are currently running"
        )

    def __init__(self):
        super().__init__(
            id="47cc3cd8-840f-4ec4-8d40-fcaba75fbe1a",
            description="Get a quick status overview of a webset",
            categories={BlockCategory.SEARCH},
            input_schema=ExaWebsetStatusBlock.Input,
            output_schema=ExaWebsetStatusBlock.Output,
            effect=BlockEffect.READ,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "webset_id": WEBSET_ID},
            test_output=[
                ("webset_id", WEBSET_ID),
                ("status", "pending"),
                ("item_count", 5),
                ("search_count", 1),
                ("enrichment_count", 0),
                ("monitor_count", 0),
                ("last_updated", "2024-01-01T00:00:00Z"),
                ("is_processing", True),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_api": lambda *args, **kwargs: sample_webset(status="pending")},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        webset = await self._get_webset(credentials, input_data.webset_id)

        yield "webset_id", webset.id
        yield "status", webset.status
        yield "item_count", webset.items_found
        yield "search_count", len(webset.searches)
        yield "enrichment_count", len(webset.enrichments)
        yield "monitor_count", len(webset.monitors)
        yield "last_updated", webset.updated_at or ""
        yield "is_processing", webset.is_processing


class ExaWebsetSummaryBlock(ExaWebsetsBlockBase):
    """Get a comprehensive summary of a webset including samples and statistics."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        include_sample_items: bool = SchemaField(
            default=True,
            description="Include sample items in the summary",
        )
        sample_size: int = SchemaField(
            default=3,
            description="Number of sample items to include",
            ge=0,
            le=10,
        )
        include_search_details: bool = SchemaField(
            default=True,
            description="Include details about searches",
        )
        include_enrichment_details: bool = SchemaField(
            default=True,
            description="Include details about enrichments",
        )

    class Output(BlockSchemaOutput):
        webset_id: str = SchemaField(description="The webset identifier")
        status: str = SchemaField(description="Current status")
        entity_type: str = SchemaField(description="Type of entities in the webset")
        total_items: int = SchemaField(description="Total number of items")
        sample_items: list[dict[str, Any]] = SchemaField(
            description="Sample items from the webset"
        )
        search_summary: SearchSummaryModel = SchemaField(
            description="Summary of searches performed"
        )
        enrichment_summary: EnrichmentSummaryModel = SchemaField(
            description="Summary of enrichments applied"
        )
        monitor_summary: MonitorSummaryModel = SchemaField(
            description="Summary of monitors configured"
        )
        statistics: WebsetStatisticsModel = SchemaField(
            description="Various statistics about the webset"
        )
        created_at: str = SchemaField(description="When the webset was created")
        updated_at: str = SchemaField(description="When the webset was last updated")

    def __init__(self):
        super().__init__(
            id="9eff1710-a49b-490e-b486-197bf8b23c61",
            description="Get a comprehensive summary of a webset with samples and statistics",
            categories={BlockCategory.SEARCH},
            input_schema=ExaWebsetSummaryBlock.Input,
            output_schema=ExaWebsetSummaryBlock.Output,
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        webset = await self._get_webset(credentials, input_data.webset_id)

        sample_items: list[dict[str, Any]] = []
        if input_data.include_sample_items and input_data.sample_size > 0:
            response = await self._api(
                credentials,
                "GET",
                f"/websets/{segment(input_data.webset_id)}/items",
                params={"limit": input_data.sample_size},
            )
            sample_items = response.get("data", [])

        search_summary = _search_summary(webset, input_data.include_search_details)
        yield "webset_id", webset.id
        yield "status", webset.status
        yield "entity_type", webset.first_entity_type or "unknown"
        yield "total_items", len(sample_items)
        yield "sample_items", sample_items
        yield "search_summary", search_summary
        yield (
            "enrichment_summary",
            _enrichment_summary(webset, input_data.include_enrichment_details),
        )
        yield "monitor_summary", _monitor_summary(webset)
        yield "statistics", _statistics(webset, search_summary)
        yield "created_at", webset.created_at or ""
        yield "updated_at", webset.updated_at or ""


def _search_summary(webset: ApiWebset, include: bool) -> SearchSummaryModel:
    searches = webset.searches if include else []
    return SearchSummaryModel(
        total_searches=len(searches),
        completed_searches=sum(1 for s in searches if s.status == "completed"),
        total_items_found=sum(s.found for s in searches),
        queries=[s.query for s in searches[:3]],
    )


def _enrichment_summary(webset: ApiWebset, include: bool) -> EnrichmentSummaryModel:
    enrichments = webset.enrichments if include else []
    return EnrichmentSummaryModel(
        total_enrichments=len(enrichments),
        completed_enrichments=sum(1 for e in enrichments if e.status == "completed"),
        enrichment_types=list({e.format or "text" for e in enrichments}),
        titles=[(e.title or e.description or "")[:50] for e in enrichments[:3]],
    )


def _monitor_summary(webset: ApiWebset) -> MonitorSummaryModel:
    next_runs = [m.next_run_at for m in webset.monitors if m.next_run_at]
    return MonitorSummaryModel(
        total_monitors=len(webset.monitors),
        active_monitors=sum(1 for m in webset.monitors if m.status == "enabled"),
        next_run=datetime.fromisoformat(min(next_runs)) if next_runs else None,
    )


def _statistics(
    webset: ApiWebset, search_summary: SearchSummaryModel
) -> WebsetStatisticsModel:
    searches = len(webset.searches)
    return WebsetStatisticsModel(
        total_operations=searches + len(webset.enrichments),
        is_processing=webset.is_processing,
        has_monitors=bool(webset.monitors),
        avg_items_per_search=(
            search_summary.total_items_found / searches if searches else 0
        ),
    )


class ExaWebsetReadyCheckBlock(ExaWebsetsBlockBase):
    """Check if a webset is ready for the next operation (conditional workflow helper)."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset to check",
            placeholder="webset-id-or-external-id",
        )
        min_items: int = SchemaField(
            default=1,
            description="Minimum number of items required to be 'ready'",
            ge=0,
        )

    class Output(BlockSchemaOutput):
        is_ready: bool = SchemaField(
            description="True if webset is idle AND has minimum items"
        )
        status: str = SchemaField(description="Current webset status")
        item_count: int = SchemaField(description="Number of items in webset")
        has_searches: bool = SchemaField(
            description="Whether webset has any searches configured"
        )
        has_enrichments: bool = SchemaField(
            description="Whether webset has any enrichments"
        )
        recommendation: str = SchemaField(
            description="Suggested next action (ready_to_process, waiting_for_results, needs_search, etc.)"
        )

    def __init__(self):
        super().__init__(
            id="faf9f0f3-e659-4264-b33b-284a02166bec",
            description="Check if webset is ready for next operation - enables conditional workflow branching",
            categories={BlockCategory.SEARCH, BlockCategory.LOGIC},
            input_schema=ExaWebsetReadyCheckBlock.Input,
            output_schema=ExaWebsetReadyCheckBlock.Output,
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        webset = await self._get_webset(credentials, input_data.webset_id)
        item_count = webset.items_found
        has_min_items = item_count >= input_data.min_items

        yield "is_ready", webset.status == "idle" and has_min_items
        yield "status", webset.status
        yield "item_count", item_count
        yield "has_searches", bool(webset.searches)
        yield "has_enrichments", bool(webset.enrichments)
        yield "recommendation", _recommendation(webset, has_min_items)


def _recommendation(webset: ApiWebset, has_min_items: bool) -> str:
    if not webset.searches:
        return "needs_search"
    if webset.is_processing:
        return "waiting_for_results"
    if not has_min_items:
        return "insufficient_items"
    if not webset.enrichments:
        return "ready_to_enrich"
    return "ready_to_process"
