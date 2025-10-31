import time
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from exa_py import AsyncExa, Exa
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
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
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
    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
        webset: Webset = SchemaField(description="The created webset with full details")
        initial_item_count: Optional[int] = SchemaField(
            description="Number of items found in the initial search (only if wait_for_initial_results was True)"
        )
        completion_time: Optional[float] = SchemaField(
            description="Time taken to complete the initial search in seconds (only if wait_for_initial_results was True)"
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

        criteria = None
        if input_data.search_criteria:
            criteria = [
                CreateCriterionParameters(description=item)
                for item in input_data.search_criteria
            ]

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

            webset_result = Webset.model_validate(webset.model_dump(by_alias=True))

            # If wait_for_initial_results is True, poll for completion
            if input_data.wait_for_initial_results and search_params:
                final_webset = exa.websets.wait_until_idle(
                    id=webset_result.id,
                    timeout=input_data.polling_timeout,
                    poll_interval=5,
                )
                completion_time = time.time() - start_time

                item_count = 0
                if final_webset.searches:
                    for search in final_webset.searches:
                        if search.progress:
                            item_count += search.progress.found

                yield "webset", webset_result
                yield "initial_item_count", item_count
                yield "completion_time", completion_time
            else:
                yield "webset", webset_result

        except ValueError as e:
            raise ValueError(f"Invalid webset configuration: {e}") from e


class ExaCreateOrFindWebsetBlock(Block):
    """Create a new webset or return existing one if external_id already exists (idempotent)."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )

        external_id: str = SchemaField(
            description="External identifier for this webset - used to find existing or create new",
            placeholder="my-unique-webset-id",
        )

        search_query: Optional[str] = SchemaField(
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

        metadata: Optional[dict] = SchemaField(
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
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        import httpx

        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        try:
            webset = aexa.websets.get(id=input_data.external_id)
            webset_result = Webset.model_validate(webset.model_dump(by_alias=True))

            yield "webset", webset_result
            yield "was_created", False

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Not found - create new webset
                search_params = None
                if input_data.search_query:
                    search_params = CreateWebsetParametersSearch(
                        query=input_data.search_query,
                        count=input_data.search_count,
                    )

                webset = aexa.websets.create(
                    params=CreateWebsetParameters(
                        search=search_params,
                        external_id=input_data.external_id,
                        metadata=input_data.metadata,
                    )
                )

                webset_result = Webset.model_validate(webset.model_dump(by_alias=True))

                yield "webset", webset_result
                yield "was_created", True
            else:
                # Other HTTP errors should propagate
                raise


class ExaUpdateWebsetBlock(Block):
    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
        webset_id: str = SchemaField(description="The unique identifier for the webset")
        status: str = SchemaField(description="The status of the webset")
        external_id: Optional[str] = SchemaField(
            description="The external identifier for the webset"
        )
        metadata: dict = SchemaField(description="Updated metadata for the webset")
        updated_at: str = SchemaField(
            description="The date and time the webset was updated"
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
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        payload = {}
        if input_data.metadata is not None:
            payload["metadata"] = input_data.metadata

        sdk_webset = aexa.websets.update(id=input_data.webset_id, params=payload)

        status_str = (
            sdk_webset.status.value
            if hasattr(sdk_webset.status, "value")
            else str(sdk_webset.status)
        )

        yield "webset_id", sdk_webset.id
        yield "status", status_str
        yield "external_id", sdk_webset.external_id
        yield "metadata", sdk_webset.metadata or {}
        yield "updated_at", (
            sdk_webset.updated_at.isoformat() if sdk_webset.updated_at else ""
        )


class ExaListWebsetsBlock(Block):
    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
        websets: list[Webset] = SchemaField(description="List of websets")
        has_more: bool = SchemaField(
            description="Whether there are more results to paginate through"
        )
        next_cursor: Optional[str] = SchemaField(
            description="Cursor for the next page of results"
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
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        response = aexa.websets.list(
            cursor=input_data.cursor,
            limit=input_data.limit,
        )

        websets_data = [
            w.model_dump(by_alias=True, exclude_none=True) for w in response.data
        ]

        yield "websets", websets_data
        yield "has_more", response.has_more
        yield "next_cursor", response.next_cursor


class ExaGetWebsetBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset to retrieve",
            placeholder="webset-id-or-external-id",
        )

    class Output(BlockSchemaOutput):
        webset_id: str = SchemaField(description="The unique identifier for the webset")
        status: str = SchemaField(description="The status of the webset")
        external_id: Optional[str] = SchemaField(
            description="The external identifier for the webset"
        )
        searches: list[dict] = SchemaField(
            description="The searches performed on the webset"
        )
        enrichments: list[dict] = SchemaField(
            description="The enrichments applied to the webset"
        )
        monitors: list[dict] = SchemaField(description="The monitors for the webset")
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
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        sdk_webset = aexa.websets.get(id=input_data.webset_id)

        status_str = (
            sdk_webset.status.value
            if hasattr(sdk_webset.status, "value")
            else str(sdk_webset.status)
        )

        searches_data = [
            s.model_dump(by_alias=True, exclude_none=True)
            for s in sdk_webset.searches or []
        ]
        enrichments_data = [
            e.model_dump(by_alias=True, exclude_none=True)
            for e in sdk_webset.enrichments or []
        ]
        monitors_data = [
            m.model_dump(by_alias=True, exclude_none=True)
            for m in sdk_webset.monitors or []
        ]

        yield "webset_id", sdk_webset.id
        yield "status", status_str
        yield "external_id", sdk_webset.external_id
        yield "searches", searches_data
        yield "enrichments", enrichments_data
        yield "monitors", monitors_data
        yield "metadata", sdk_webset.metadata or {}
        yield "created_at", (
            sdk_webset.created_at.isoformat() if sdk_webset.created_at else ""
        )
        yield "updated_at", (
            sdk_webset.updated_at.isoformat() if sdk_webset.updated_at else ""
        )


class ExaDeleteWebsetBlock(Block):
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
        external_id: Optional[str] = SchemaField(
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
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        deleted_webset = aexa.websets.delete(id=input_data.webset_id)

        status_str = (
            deleted_webset.status.value
            if hasattr(deleted_webset.status, "value")
            else str(deleted_webset.status)
        )

        yield "webset_id", deleted_webset.id
        yield "external_id", deleted_webset.external_id
        yield "status", status_str
        yield "success", "true"


class ExaCancelWebsetBlock(Block):
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
        external_id: Optional[str] = SchemaField(
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
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        canceled_webset = aexa.websets.cancel(id=input_data.webset_id)

        status_str = (
            canceled_webset.status.value
            if hasattr(canceled_webset.status, "value")
            else str(canceled_webset.status)
        )

        yield "webset_id", canceled_webset.id
        yield "status", status_str
        yield "external_id", canceled_webset.external_id
        yield "success", "true"


# Mirrored models for Preview response stability
class PreviewCriterionModel(BaseModel):
    """Stable model for preview criteria."""

    description: str

    @classmethod
    def from_sdk(cls, sdk_criterion) -> "PreviewCriterionModel":
        """Convert SDK criterion to our model."""
        return cls(description=sdk_criterion.description)


class PreviewEnrichmentModel(BaseModel):
    """Stable model for preview enrichment."""

    description: str
    format: str
    options: List[str]

    @classmethod
    def from_sdk(cls, sdk_enrichment) -> "PreviewEnrichmentModel":
        """Convert SDK enrichment to our model."""
        format_str = (
            sdk_enrichment.format.value
            if hasattr(sdk_enrichment.format, "value")
            else str(sdk_enrichment.format)
        )

        options_list = []
        if sdk_enrichment.options:
            for opt in sdk_enrichment.options:
                opt_dict = opt.model_dump(by_alias=True)
                options_list.append(opt_dict.get("label", ""))

        return cls(
            description=sdk_enrichment.description,
            format=format_str,
            options=options_list,
        )


class PreviewSearchModel(BaseModel):
    """Stable model for preview search details."""

    entity_type: str
    entity_description: Optional[str]
    criteria: List[PreviewCriterionModel]

    @classmethod
    def from_sdk(cls, sdk_search) -> "PreviewSearchModel":
        """Convert SDK search preview to our model."""
        # Extract entity type from union
        entity_dict = sdk_search.entity.model_dump(by_alias=True)
        entity_type = entity_dict.get("type", "auto")
        entity_description = entity_dict.get("description")

        # Convert criteria
        criteria = [
            PreviewCriterionModel.from_sdk(c) for c in sdk_search.criteria or []
        ]

        return cls(
            entity_type=entity_type,
            entity_description=entity_description,
            criteria=criteria,
        )


class PreviewWebsetModel(BaseModel):
    """Stable model for preview response."""

    search: PreviewSearchModel
    enrichments: List[PreviewEnrichmentModel]

    @classmethod
    def from_sdk(cls, sdk_preview) -> "PreviewWebsetModel":
        """Convert SDK PreviewWebsetResponse to our model."""

        search = PreviewSearchModel.from_sdk(sdk_preview.search)
        enrichments = [
            PreviewEnrichmentModel.from_sdk(e) for e in sdk_preview.enrichments or []
        ]

        return cls(search=search, enrichments=enrichments)


class ExaPreviewWebsetBlock(Block):
    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
        preview: PreviewWebsetModel = SchemaField(
            description="Full preview response with search and enrichment details"
        )
        entity_type: str = SchemaField(
            description="The detected or specified entity type"
        )
        entity_description: Optional[str] = SchemaField(
            description="Description of the entity type"
        )
        criteria: list[PreviewCriterionModel] = SchemaField(
            description="Generated search criteria that will be used"
        )
        enrichment_columns: list[PreviewEnrichmentModel] = SchemaField(
            description="Available enrichment columns that can be extracted"
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
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        payload: dict[str, Any] = {
            "query": input_data.query,
        }

        if input_data.entity_type:
            entity: dict[str, Any] = {"type": input_data.entity_type.value}
            if (
                input_data.entity_type == SearchEntityType.CUSTOM
                and input_data.entity_description
            ):
                entity["description"] = input_data.entity_description
            payload["entity"] = entity

        sdk_preview = aexa.websets.preview(params=payload)

        preview = PreviewWebsetModel.from_sdk(sdk_preview)

        entity_type = preview.search.entity_type
        entity_description = preview.search.entity_description
        criteria = preview.search.criteria
        enrichments = preview.enrichments

        # Generate interpretation
        interpretation = f"Query will search for {entity_type}"
        if entity_description:
            interpretation += f" ({entity_description})"
        if criteria:
            interpretation += f" with {len(criteria)} criteria"
        if enrichments:
            interpretation += f" and {len(enrichments)} available enrichment columns"

        # Generate suggestions
        suggestions = []
        if not criteria:
            suggestions.append(
                "Consider adding specific criteria to narrow your search"
            )
        if not enrichments:
            suggestions.append(
                "Consider specifying what data points you want to extract"
            )

        # Yield full model first
        yield "preview", preview

        # Then yield individual fields for graph flexibility
        yield "entity_type", entity_type
        yield "entity_description", entity_description
        yield "criteria", criteria
        yield "enrichment_columns", enrichments
        yield "interpretation", interpretation
        yield "suggestions", suggestions


class ExaWebsetStatusBlock(Block):
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
            description="Current status (idle, running, paused, etc.)"
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
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        webset = aexa.websets.get(id=input_data.webset_id)

        status = (
            webset.status.value
            if hasattr(webset.status, "value")
            else str(webset.status)
        )
        is_processing = status in ["running", "pending"]

        # Estimate item count from search progress
        item_count = 0
        if webset.searches:
            for search in webset.searches:
                if search.progress:
                    item_count += search.progress.found

        # Count searches, enrichments, monitors
        search_count = len(webset.searches or [])
        enrichment_count = len(webset.enrichments or [])
        monitor_count = len(webset.monitors or [])

        yield "webset_id", webset.id
        yield "status", status
        yield "item_count", item_count
        yield "search_count", search_count
        yield "enrichment_count", enrichment_count
        yield "monitor_count", monitor_count
        yield "last_updated", webset.updated_at.isoformat() if webset.updated_at else ""
        yield "is_processing", is_processing


# Summary models for ExaWebsetSummaryBlock
class SearchSummaryModel(BaseModel):
    """Summary of searches in a webset."""

    total_searches: int
    completed_searches: int
    total_items_found: int
    queries: List[str]


class EnrichmentSummaryModel(BaseModel):
    """Summary of enrichments in a webset."""

    total_enrichments: int
    completed_enrichments: int
    enrichment_types: List[str]
    titles: List[str]


class MonitorSummaryModel(BaseModel):
    """Summary of monitors in a webset."""

    total_monitors: int
    active_monitors: int
    next_run: Optional[datetime] = None


class WebsetStatisticsModel(BaseModel):
    """Various statistics about a webset."""

    total_operations: int
    is_processing: bool
    has_monitors: bool
    avg_items_per_search: float


class ExaWebsetSummaryBlock(Block):
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
        sample_items: list[Dict[str, Any]] = SchemaField(
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
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        webset = aexa.websets.get(id=input_data.webset_id)

        # Extract basic info
        webset_id = webset.id
        status = (
            webset.status.value
            if hasattr(webset.status, "value")
            else str(webset.status)
        )

        # Determine entity type from searches
        entity_type = "unknown"
        searches = webset.searches or []
        if searches:
            first_search = searches[0]
            if first_search.entity:
                entity_dict = first_search.entity.model_dump(
                    by_alias=True, exclude_none=True
                )
                entity_type = entity_dict.get("type", "unknown")

        # Get sample items if requested
        sample_items_data = []
        total_items = 0

        if input_data.include_sample_items and input_data.sample_size > 0:
            items_response = aexa.websets.items.list(
                webset_id=input_data.webset_id, limit=input_data.sample_size
            )
            sample_items_data = [
                item.model_dump(by_alias=True, exclude_none=True)
                for item in items_response.data
            ]
            total_items = len(sample_items_data)

        # Build search summary using Pydantic model
        search_summary = SearchSummaryModel(
            total_searches=0,
            completed_searches=0,
            total_items_found=0,
            queries=[],
        )
        if input_data.include_search_details and searches:
            search_summary = SearchSummaryModel(
                total_searches=len(searches),
                completed_searches=sum(
                    1
                    for s in searches
                    if (s.status.value if hasattr(s.status, "value") else str(s.status))
                    == "completed"
                ),
                total_items_found=int(
                    sum(s.progress.found if s.progress else 0 for s in searches)
                ),
                queries=[s.query for s in searches[:3]],  # First 3 queries
            )

        # Build enrichment summary using Pydantic model
        enrichment_summary = EnrichmentSummaryModel(
            total_enrichments=0,
            completed_enrichments=0,
            enrichment_types=[],
            titles=[],
        )
        enrichments = webset.enrichments or []
        if input_data.include_enrichment_details and enrichments:
            enrichment_summary = EnrichmentSummaryModel(
                total_enrichments=len(enrichments),
                completed_enrichments=sum(
                    1
                    for e in enrichments
                    if (e.status.value if hasattr(e.status, "value") else str(e.status))
                    == "completed"
                ),
                enrichment_types=list(
                    set(
                        (
                            e.format.value
                            if e.format and hasattr(e.format, "value")
                            else str(e.format) if e.format else "text"
                        )
                        for e in enrichments
                    )
                ),
                titles=[(e.title or e.description or "")[:50] for e in enrichments[:3]],
            )

        # Build monitor summary using Pydantic model
        monitors = webset.monitors or []
        next_run_dt = None
        if monitors:
            next_runs = [m.next_run_at for m in monitors if m.next_run_at]
            if next_runs:
                next_run_dt = min(next_runs)

        monitor_summary = MonitorSummaryModel(
            total_monitors=len(monitors),
            active_monitors=sum(
                1
                for m in monitors
                if (m.status.value if hasattr(m.status, "value") else str(m.status))
                == "enabled"
            ),
            next_run=next_run_dt,
        )

        # Build statistics using Pydantic model
        statistics = WebsetStatisticsModel(
            total_operations=len(searches) + len(enrichments),
            is_processing=status in ["running", "pending"],
            has_monitors=len(monitors) > 0,
            avg_items_per_search=(
                search_summary.total_items_found / len(searches) if searches else 0
            ),
        )

        yield "webset_id", webset_id
        yield "status", status
        yield "entity_type", entity_type
        yield "total_items", total_items
        yield "sample_items", sample_items_data
        yield "search_summary", search_summary
        yield "enrichment_summary", enrichment_summary
        yield "monitor_summary", monitor_summary
        yield "statistics", statistics
        yield "created_at", webset.created_at.isoformat() if webset.created_at else ""
        yield "updated_at", webset.updated_at.isoformat() if webset.updated_at else ""


class ExaWebsetReadyCheckBlock(Block):
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
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        # Get webset details
        webset = aexa.websets.get(id=input_data.webset_id)

        status = (
            webset.status.value
            if hasattr(webset.status, "value")
            else str(webset.status)
        )

        # Estimate item count from search progress
        item_count = 0
        if webset.searches:
            for search in webset.searches:
                if search.progress:
                    item_count += search.progress.found

        # Determine readiness
        is_idle = status == "idle"
        has_min_items = item_count >= input_data.min_items
        is_ready = is_idle and has_min_items

        # Check resources
        has_searches = len(webset.searches or []) > 0
        has_enrichments = len(webset.enrichments or []) > 0

        # Generate recommendation
        recommendation = ""
        if not has_searches:
            recommendation = "needs_search"
        elif status in ["running", "pending"]:
            recommendation = "waiting_for_results"
        elif not has_min_items:
            recommendation = "insufficient_items"
        elif not has_enrichments:
            recommendation = "ready_to_enrich"
        else:
            recommendation = "ready_to_process"

        yield "is_ready", is_ready
        yield "status", status
        yield "item_count", item_count
        yield "has_searches", has_searches
        yield "has_enrichments", has_enrichments
        yield "recommendation", recommendation
