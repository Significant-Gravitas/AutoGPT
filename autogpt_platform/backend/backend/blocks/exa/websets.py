from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional
import asyncio
import time

from exa_py import Exa
from exa_py.websets.types import (
    CreateCriterionParameters,
    CreateEnrichmentParameters,
    CreateWebsetParameters,
    CreateWebsetParametersSearch,
    ExcludeItem,
    Format,
    ImportItem,
    ImportSource,
    Option,
    ScopeItem,
    ScopeRelationship,
    ScopeSourceType,
    WebsetArticleEntity,
    WebsetCompanyEntity,
    WebsetCustomEntity,
    WebsetPersonEntity,
    WebsetResearchPaperEntity,
    WebsetStatus,
)
from pydantic import Field

from backend.sdk import (
    APIKeyCredentials,
    BaseModel,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    Requests,
    SchemaField,
)

from ._config import exa


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


class Webset(BaseModel):
    id: str
    status: WebsetStatus | None = Field(..., title="WebsetStatus")
    """
    The status of the webset
    """
    external_id: Annotated[Optional[str], Field(alias="externalId")] = None
    """
    The external identifier for the webset
    NOTE: Returning dict to avoid ui crashing due to nested objects
    """
    searches: List[dict[str, Any]] | None = None
    """
    The searches that have been performed on the webset.
    NOTE: Returning dict to avoid ui crashing due to nested objects
    """
    enrichments: List[dict[str, Any]] | None = None
    """
    The Enrichments to apply to the Webset Items.
    NOTE: Returning dict to avoid ui crashing due to nested objects
    """
    monitors: List[dict[str, Any]] | None = None
    """
    The Monitors for the Webset.
    NOTE: Returning dict to avoid ui crashing due to nested objects
    """
    metadata: Optional[Dict[str, Any]] = {}
    """
    Set of key-value pairs you want to associate with this object.
    """
    created_at: Annotated[datetime | None, Field(alias="createdAt")] = None
    """
    The date and time the webset was created
    """
    updated_at: Annotated[datetime | None, Field(alias="updatedAt")] = None
    """
    The date and time the webset was last updated
    """


class ExaCreateWebsetBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )

        # Search parameters (flattened)
        search_query: str = SchemaField(
            description="Your search query. Use this to describe what you are looking for. Any URL provided will be crawled and used as context for the search.",
            placeholder="Marketing agencies based in the US, that focus on consumer products",
        )
        search_count: Optional[int] = SchemaField(
            default=10,
            description="Number of items the search will attempt to find. The actual number of items found may be less than this number depending on the search complexity.",
            ge=1,
            le=1000,
        )
        search_entity_type: SearchEntityType = SchemaField(
            default=SearchEntityType.AUTO,
            description="Entity type: 'company', 'person', 'article', 'research_paper', or 'custom'. If not provided, we automatically detect the entity from the query.",
            advanced=True,
        )
        search_entity_description: Optional[str] = SchemaField(
            default=None,
            description="Description for custom entity type (required when search_entity_type is 'custom')",
            advanced=True,
        )

        # Search criteria (flattened)
        search_criteria: list[str] = SchemaField(
            default_factory=list,
            description="List of criteria descriptions that every item will be evaluated against. If not provided, we automatically detect the criteria from the query.",
            advanced=True,
        )

        # Search exclude sources (flattened)
        search_exclude_sources: list[str] = SchemaField(
            default_factory=list,
            description="List of source IDs (imports or websets) to exclude from search results",
            advanced=True,
        )
        search_exclude_types: list[SearchType] = SchemaField(
            default_factory=list,
            description="List of source types corresponding to exclude sources ('import' or 'webset')",
            advanced=True,
        )

        # Search scope sources (flattened)
        search_scope_sources: list[str] = SchemaField(
            default_factory=list,
            description="List of source IDs (imports or websets) to limit search scope to",
            advanced=True,
        )
        search_scope_types: list[SearchType] = SchemaField(
            default_factory=list,
            description="List of source types corresponding to scope sources ('import' or 'webset')",
            advanced=True,
        )
        search_scope_relationships: list[str] = SchemaField(
            default_factory=list,
            description="List of relationship definitions for hop searches (optional, one per scope source)",
            advanced=True,
        )
        search_scope_relationship_limits: list[int] = SchemaField(
            default_factory=list,
            description="List of limits on the number of related entities to find (optional, one per scope relationship)",
            advanced=True,
        )

        # Import parameters (flattened)
        import_sources: list[str] = SchemaField(
            default_factory=list,
            description="List of source IDs to import from",
            advanced=True,
        )
        import_types: list[SearchType] = SchemaField(
            default_factory=list,
            description="List of source types corresponding to import sources ('import' or 'webset')",
            advanced=True,
        )

        # Enrichment parameters (flattened)
        enrichment_descriptions: list[str] = SchemaField(
            default_factory=list,
            description="List of enrichment task descriptions to perform on each webset item",
            advanced=True,
        )
        enrichment_formats: list[EnrichmentFormat] = SchemaField(
            default_factory=list,
            description="List of formats for enrichment responses ('text', 'date', 'number', 'options', 'email', 'phone'). If not specified, we automatically select the best format.",
            advanced=True,
        )
        enrichment_options: list[list[str]] = SchemaField(
            default_factory=list,
            description="List of option lists for enrichments with 'options' format. Each inner list contains the option labels.",
            advanced=True,
        )
        enrichment_metadata: list[dict] = SchemaField(
            default_factory=list,
            description="List of metadata dictionaries for enrichments",
            advanced=True,
        )

        # Webset metadata
        external_id: Optional[str] = SchemaField(
            default=None,
            description="External identifier for the webset. You can use this to reference the webset by your own internal identifiers.",
            placeholder="my-webset-123",
            advanced=True,
        )
        metadata: Optional[dict] = SchemaField(
            default_factory=dict,
            description="Key-value pairs to associate with this webset",
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

    class Output(BlockSchema):
        webset: Webset = SchemaField(
            description="The created webset with full details"
        )
        initial_item_count: Optional[int] = SchemaField(
            description="Number of items found in the initial search (only if wait_for_initial_results was True)",
            default=None,
        )
        completion_time: Optional[float] = SchemaField(
            description="Time taken to complete the initial search in seconds (only if wait_for_initial_results was True)",
            default=None,
        )
        error: str = SchemaField(
            description="Error message if the operation failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="0cda29ff-c549-4a19-8805-c982b7d4ec34",
            description="Create a new Exa Webset for persistent web search collections with optional waiting for initial results",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCreateWebsetBlock.Input,
            output_schema=ExaCreateWebsetBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:

        exa = Exa(credentials.api_key.get_secret_value())

        # ------------------------------------------------------------
        # Build entity (if explicitly provided)
        # ------------------------------------------------------------
        entity = None
        if input_data.search_entity_type == SearchEntityType.COMPANY:
            entity = WebsetCompanyEntity(type="company")
        elif input_data.search_entity_type == SearchEntityType.PERSON:
            entity = WebsetPersonEntity(type="person")
        elif input_data.search_entity_type == SearchEntityType.ARTICLE:
            entity = WebsetArticleEntity(type="article")
        elif input_data.search_entity_type == SearchEntityType.RESEARCH_PAPER:
            entity = WebsetResearchPaperEntity(type="research_paper")
        elif (
            input_data.search_entity_type == SearchEntityType.CUSTOM
            and input_data.search_entity_description
        ):
            entity = WebsetCustomEntity(
                type="custom", description=input_data.search_entity_description
            )

        # ------------------------------------------------------------
        # Build criteria list
        # ------------------------------------------------------------
        criteria = None
        if input_data.search_criteria:
            criteria = [
                CreateCriterionParameters(description=item)
                for item in input_data.search_criteria
            ]

        # ------------------------------------------------------------
        # Build exclude sources list
        # ------------------------------------------------------------
        exclude_items = None
        if input_data.search_exclude_sources:
            exclude_items = []
            for idx, src_id in enumerate(input_data.search_exclude_sources):
                src_type = None
                if input_data.search_exclude_types and idx < len(
                    input_data.search_exclude_types
                ):
                    src_type = input_data.search_exclude_types[idx]
                # Default to IMPORT if type missing
                if src_type == SearchType.WEBSET:
                    source_enum = ImportSource.webset
                else:
                    source_enum = ImportSource.import_
                exclude_items.append(ExcludeItem(source=source_enum, id=src_id))

        # ------------------------------------------------------------
        # Build scope list
        # ------------------------------------------------------------
        scope_items = None
        if input_data.search_scope_sources:
            scope_items = []
            for idx, src_id in enumerate(input_data.search_scope_sources):
                src_type = None
                if input_data.search_scope_types and idx < len(
                    input_data.search_scope_types
                ):
                    src_type = input_data.search_scope_types[idx]
                relationship = None
                if input_data.search_scope_relationships and idx < len(
                    input_data.search_scope_relationships
                ):
                    rel_def = input_data.search_scope_relationships[idx]
                    lim = None
                    if input_data.search_scope_relationship_limits and idx < len(
                        input_data.search_scope_relationship_limits
                    ):
                        lim = input_data.search_scope_relationship_limits[idx]
                    relationship = ScopeRelationship(definition=rel_def, limit=lim)
                if src_type == SearchType.WEBSET:
                    src_enum = ScopeSourceType.webset
                else:
                    src_enum = ScopeSourceType.import_
                scope_items.append(
                    ScopeItem(source=src_enum, id=src_id, relationship=relationship)
                )

        # ------------------------------------------------------------
        # Assemble search parameters (only if a query is provided)
        # ------------------------------------------------------------
        search_params = None
        if input_data.search_query:
            search_params = CreateWebsetParametersSearch(
                query=input_data.search_query,
                count=input_data.search_count,
                entity=entity,
                criteria=criteria,
                exclude=exclude_items,
                scope=scope_items,
            )

        # ------------------------------------------------------------
        # Build imports list
        # ------------------------------------------------------------
        imports_params = None
        if input_data.import_sources:
            imports_params = []
            for idx, src_id in enumerate(input_data.import_sources):
                src_type = None
                if input_data.import_types and idx < len(input_data.import_types):
                    src_type = input_data.import_types[idx]
                if src_type == SearchType.WEBSET:
                    source_enum = ImportSource.webset
                else:
                    source_enum = ImportSource.import_
                imports_params.append(ImportItem(source=source_enum, id=src_id))

        # ------------------------------------------------------------
        # Build enrichment list
        # ------------------------------------------------------------
        enrichments_params = None
        if input_data.enrichment_descriptions:
            enrichments_params = []
            for idx, desc in enumerate(input_data.enrichment_descriptions):
                fmt = None
                if input_data.enrichment_formats and idx < len(
                    input_data.enrichment_formats
                ):
                    fmt_enum = input_data.enrichment_formats[idx]
                    if fmt_enum is not None:
                        fmt = Format(
                            fmt_enum.value if isinstance(fmt_enum, Enum) else fmt_enum
                        )
                options_list = None
                if input_data.enrichment_options and idx < len(
                    input_data.enrichment_options
                ):
                    raw_opts = input_data.enrichment_options[idx]
                    if raw_opts:
                        options_list = [Option(label=o) for o in raw_opts]
                metadata_obj = None
                if input_data.enrichment_metadata and idx < len(
                    input_data.enrichment_metadata
                ):
                    metadata_obj = input_data.enrichment_metadata[idx]
                enrichments_params.append(
                    CreateEnrichmentParameters(
                        description=desc,
                        format=fmt,
                        options=options_list,
                        metadata=metadata_obj,
                    )
                )

        # ------------------------------------------------------------
        # Create the webset
        # ------------------------------------------------------------
        try:
            start_time = time.time()
            webset = exa.websets.create(
                params=CreateWebsetParameters(
                    search=search_params,
                    imports=imports_params,
                    enrichments=enrichments_params,
                    external_id=input_data.external_id,
                    metadata=input_data.metadata,
                )
            )

            # Convert to our Webset model
            webset_result = Webset.model_validate(webset.model_dump(by_alias=True))

            # If wait_for_initial_results is True, poll for completion
            if input_data.wait_for_initial_results and search_params:
                item_count = await self._poll_for_completion(
                    webset_result.id,
                    credentials.api_key.get_secret_value(),
                    input_data.polling_timeout
                )
                completion_time = time.time() - start_time

                yield "webset", webset_result
                yield "initial_item_count", item_count
                yield "completion_time", completion_time
            else:
                yield "webset", webset_result

        except Exception as e:
            yield "error", str(e)
            yield "webset", Webset(id="", status=None)

    async def _poll_for_completion(
        self, webset_id: str, api_key: str, timeout: int
    ) -> int:
        """Poll webset status until it becomes idle or times out."""
        start_time = time.time()
        interval = 5  # Start with 5 second intervals
        max_interval = 30  # Cap at 30 seconds

        url = f"https://api.exa.ai/websets/v0/websets/{webset_id}"
        headers = {
            "x-api-key": api_key,
        }

        while time.time() - start_time < timeout:
            try:
                response = await Requests().get(url, headers=headers)
                data = response.json()

                status = data.get("status", "")

                # Check if status is idle (search complete)
                if status == "idle":
                    # Count items
                    items_url = f"https://api.exa.ai/websets/v0/websets/{webset_id}/items"
                    items_response = await Requests().get(items_url, headers=headers, params={"limit": 1})
                    items_data = items_response.json()

                    # Get total count from pagination metadata
                    total_count = len(items_data.get("data", []))

                    # If there's pagination info, use that for more accurate count
                    if "pagination" in items_data:
                        total_count = items_data["pagination"].get("total", total_count)

                    return total_count

                # Wait before next poll with exponential backoff
                await asyncio.sleep(interval)
                interval = min(interval * 1.5, max_interval)

            except Exception:
                # Continue polling on errors
                await asyncio.sleep(interval)

        # Timeout reached, return whatever we have
        return 0


class ExaUpdateWebsetBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset to update",
            placeholder="webset-id-or-external-id",
        )
        metadata: Optional[dict] = SchemaField(
            default=None,
            description="Key-value pairs to associate with this webset (set to null to clear)",
        )

    class Output(BlockSchema):
        webset_id: str = SchemaField(description="The unique identifier for the webset")
        status: str = SchemaField(description="The status of the webset")
        external_id: Optional[str] = SchemaField(
            description="The external identifier for the webset", default=None
        )
        metadata: dict = SchemaField(
            description="Updated metadata for the webset", default_factory=dict
        )
        updated_at: str = SchemaField(
            description="The date and time the webset was updated"
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="89ccd99a-3c2b-4fbf-9e25-0ffa398d0314",
            description="Update metadata for an existing Webset",
            categories={BlockCategory.SEARCH},
            input_schema=ExaUpdateWebsetBlock.Input,
            output_schema=ExaUpdateWebsetBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        # Build the payload
        payload = {}
        if input_data.metadata is not None:
            payload["metadata"] = input_data.metadata

        try:
            response = await Requests().post(url, headers=headers, json=payload)
            data = response.json()

            yield "webset_id", data.get("id", "")
            yield "status", data.get("status", "")
            yield "external_id", data.get("externalId")
            yield "metadata", data.get("metadata", {})
            yield "updated_at", data.get("updatedAt", "")

        except Exception as e:
            yield "error", str(e)
            yield "webset_id", ""
            yield "status", ""
            yield "metadata", {}
            yield "updated_at", ""


class ExaListWebsetsBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        trigger: Any | None = SchemaField(
            default=None,
            description="Trigger for the webset, value is ignored!",
            advanced=False,
        )
        cursor: Optional[str] = SchemaField(
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

    class Output(BlockSchema):
        websets: list[Webset] = SchemaField(
            description="List of websets", default_factory=list
        )
        has_more: bool = SchemaField(
            description="Whether there are more results to paginate through",
            default=False,
        )
        next_cursor: Optional[str] = SchemaField(
            description="Cursor for the next page of results", default=None
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="1dcd8fd6-c13f-4e6f-bd4c-654428fa4757",
            description="List all Websets with pagination support",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListWebsetsBlock.Input,
            output_schema=ExaListWebsetsBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = "https://api.exa.ai/websets/v0/websets"
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        params: dict[str, Any] = {
            "limit": input_data.limit,
        }
        if input_data.cursor:
            params["cursor"] = input_data.cursor

        try:
            response = await Requests().get(url, headers=headers, params=params)
            data = response.json()

            yield "websets", data.get("data", [])
            yield "has_more", data.get("hasMore", False)
            yield "next_cursor", data.get("nextCursor")

        except Exception as e:
            yield "error", str(e)
            yield "websets", []
            yield "has_more", False


class ExaGetWebsetBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset to retrieve",
            placeholder="webset-id-or-external-id",
        )

    class Output(BlockSchema):
        webset_id: str = SchemaField(description="The unique identifier for the webset")
        status: str = SchemaField(description="The status of the webset")
        external_id: Optional[str] = SchemaField(
            description="The external identifier for the webset", default=None
        )
        searches: list[dict] = SchemaField(
            description="The searches performed on the webset", default_factory=list
        )
        enrichments: list[dict] = SchemaField(
            description="The enrichments applied to the webset", default_factory=list
        )
        monitors: list[dict] = SchemaField(
            description="The monitors for the webset", default_factory=list
        )
        items: Optional[list[dict]] = SchemaField(
            description="The items in the webset (if expand_items is true)",
            default=None,
        )
        metadata: dict = SchemaField(
            description="Key-value pairs associated with the webset",
            default_factory=dict,
        )
        created_at: str = SchemaField(
            description="The date and time the webset was created"
        )
        updated_at: str = SchemaField(
            description="The date and time the webset was last updated"
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="6ab8e12a-132c-41bf-b5f3-d662620fa832",
            description="Retrieve a Webset by ID or external ID",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetWebsetBlock.Input,
            output_schema=ExaGetWebsetBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}"
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        try:
            response = await Requests().get(url, headers=headers)
            data = response.json()

            yield "webset_id", data.get("id", "")
            yield "status", data.get("status", "")
            yield "external_id", data.get("externalId")
            yield "searches", data.get("searches", [])
            yield "enrichments", data.get("enrichments", [])
            yield "monitors", data.get("monitors", [])
            yield "items", data.get("items")
            yield "metadata", data.get("metadata", {})
            yield "created_at", data.get("createdAt", "")
            yield "updated_at", data.get("updatedAt", "")

        except Exception as e:
            yield "error", str(e)
            yield "webset_id", ""
            yield "status", ""
            yield "searches", []
            yield "enrichments", []
            yield "monitors", []
            yield "metadata", {}
            yield "created_at", ""
            yield "updated_at", ""


class ExaDeleteWebsetBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset to delete",
            placeholder="webset-id-or-external-id",
        )

    class Output(BlockSchema):
        webset_id: str = SchemaField(
            description="The unique identifier for the deleted webset"
        )
        external_id: Optional[str] = SchemaField(
            description="The external identifier for the deleted webset", default=None
        )
        status: str = SchemaField(description="The status of the deleted webset")
        success: str = SchemaField(
            description="Whether the deletion was successful", default="true"
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="aa6994a2-e986-421f-8d4c-7671d3be7b7e",
            description="Delete a Webset and all its items",
            categories={BlockCategory.SEARCH},
            input_schema=ExaDeleteWebsetBlock.Input,
            output_schema=ExaDeleteWebsetBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}"
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        try:
            response = await Requests().delete(url, headers=headers)
            data = response.json()

            yield "webset_id", data.get("id", "")
            yield "external_id", data.get("externalId")
            yield "status", data.get("status", "")
            yield "success", "true"

        except Exception as e:
            yield "error", str(e)
            yield "webset_id", ""
            yield "status", ""
            yield "success", "false"


class ExaCancelWebsetBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset to cancel",
            placeholder="webset-id-or-external-id",
        )

    class Output(BlockSchema):
        webset_id: str = SchemaField(description="The unique identifier for the webset")
        status: str = SchemaField(
            description="The status of the webset after cancellation"
        )
        external_id: Optional[str] = SchemaField(
            description="The external identifier for the webset", default=None
        )
        success: str = SchemaField(
            description="Whether the cancellation was successful", default="true"
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="e40a6420-1db8-47bb-b00a-0e6aecd74176",
            description="Cancel all operations being performed on a Webset",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCancelWebsetBlock.Input,
            output_schema=ExaCancelWebsetBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/cancel"
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        try:
            response = await Requests().post(url, headers=headers)
            data = response.json()

            yield "webset_id", data.get("id", "")
            yield "status", data.get("status", "")
            yield "external_id", data.get("externalId")
            yield "success", "true"

        except Exception as e:
            yield "error", str(e)
            yield "webset_id", ""
            yield "status", ""
            yield "success", "false"


class ExaPreviewWebsetBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        query: str = SchemaField(
            description="Your search query to preview. Use this to see how Exa will interpret your search before creating a webset.",
            placeholder="Marketing agencies based in the US, with brands worked with and city",
        )
        entity_type: Optional[SearchEntityType] = SchemaField(
            default=None,
            description="Entity type to force: 'company', 'person', 'article', 'research_paper', or 'custom'. If not provided, Exa will auto-detect.",
            advanced=True,
        )
        entity_description: Optional[str] = SchemaField(
            default=None,
            description="Description for custom entity type (required when entity_type is 'custom')",
            advanced=True,
        )

    class Output(BlockSchema):
        entity_type: str = SchemaField(
            description="The detected or specified entity type"
        )
        entity_description: Optional[str] = SchemaField(
            description="Description of the entity type", default=None
        )
        criteria: list[dict] = SchemaField(
            description="Generated search criteria that will be used", default_factory=list
        )
        enrichment_columns: list[dict] = SchemaField(
            description="Available enrichment columns that can be extracted", default_factory=list
        )
        interpretation: str = SchemaField(
            description="Human-readable interpretation of how the query will be processed", default=""
        )
        suggestions: list[str] = SchemaField(
            description="Suggestions for improving the query", default_factory=list
        )
        error: str = SchemaField(
            description="Error message if the preview failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="f8c4e2a1-9b3d-4e5f-a6c7-d8e9f0a1b2c3",
            description="Preview how a search query will be interpreted before creating a webset. Helps understand entity detection, criteria generation, and available enrichments.",
            categories={BlockCategory.SEARCH},
            input_schema=ExaPreviewWebsetBlock.Input,
            output_schema=ExaPreviewWebsetBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = "https://api.exa.ai/websets/v0/websets/preview"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        # Build the payload
        payload = {
            "query": input_data.query,
        }

        # Add entity configuration if provided
        if input_data.entity_type:
            entity = {"type": input_data.entity_type.value}
            if input_data.entity_type == SearchEntityType.CUSTOM and input_data.entity_description:
                entity["description"] = input_data.entity_description
            payload["entity"] = entity

        try:
            response = await Requests().post(url, headers=headers, json=payload)
            data = response.json()

            # Extract entity information
            entity_info = data.get("entity", {})
            entity_type = entity_info.get("type", "auto")
            entity_description = entity_info.get("description")

            # Extract criteria
            criteria = data.get("criteria", [])

            # Extract enrichment columns
            enrichments = data.get("enrichmentColumns", [])

            # Generate interpretation
            interpretation = f"Query will search for {entity_type}"
            if entity_description:
                interpretation += f" ({entity_description})"
            if criteria:
                interpretation += f" with {len(criteria)} criteria"
            if enrichments:
                interpretation += f" and {len(enrichments)} available enrichment columns"

            # Generate suggestions (could be enhanced based on the response)
            suggestions = []
            if not criteria:
                suggestions.append("Consider adding specific criteria to narrow your search")
            if not enrichments:
                suggestions.append("Consider specifying what data points you want to extract")

            yield "entity_type", entity_type
            yield "entity_description", entity_description
            yield "criteria", criteria
            yield "enrichment_columns", enrichments
            yield "interpretation", interpretation
            yield "suggestions", suggestions

        except Exception as e:
            yield "error", str(e)
            yield "entity_type", ""
            yield "criteria", []
            yield "enrichment_columns", []
            yield "interpretation", ""
            yield "suggestions", []