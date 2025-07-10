from typing import Any, Optional

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    Requests,
    SchemaField,
)

from ._config import exa
from .helpers import WebsetEnrichmentConfig, WebsetSearchConfig


class ExaCreateWebsetBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        search: WebsetSearchConfig = SchemaField(
            description="Initial search configuration for the Webset"
        )
        enrichments: Optional[list[WebsetEnrichmentConfig]] = SchemaField(
            default=None,
            description="Enrichments to apply to Webset items",
            advanced=True,
        )
        external_id: Optional[str] = SchemaField(
            default=None,
            description="External identifier for the webset",
            placeholder="my-webset-123",
            advanced=True,
        )
        metadata: Optional[dict] = SchemaField(
            default=None,
            description="Key-value pairs to associate with this webset",
            advanced=True,
        )

    class Output(BlockSchema):
        webset_id: str = SchemaField(
            description="The unique identifier for the created webset"
        )
        status: str = SchemaField(description="The status of the webset")
        external_id: Optional[str] = SchemaField(
            description="The external identifier for the webset", default=None
        )
        created_at: str = SchemaField(
            description="The date and time the webset was created"
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="0cda29ff-c549-4a19-8805-c982b7d4ec34",
            description="Create a new Exa Webset for persistent web search collections",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCreateWebsetBlock.Input,
            output_schema=ExaCreateWebsetBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = "https://api.exa.ai/websets/v0/websets"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        # Build the payload
        payload: dict[str, Any] = {
            "search": input_data.search.model_dump(exclude_none=True),
        }

        # Convert enrichments to API format
        if input_data.enrichments:
            enrichments_data = []
            for enrichment in input_data.enrichments:
                enrichments_data.append(enrichment.model_dump(exclude_none=True))
            payload["enrichments"] = enrichments_data

        if input_data.external_id:
            payload["externalId"] = input_data.external_id

        if input_data.metadata:
            payload["metadata"] = input_data.metadata

        try:
            response = await Requests().post(url, headers=headers, json=payload)
            data = response.json()

            yield "webset_id", data.get("id", "")
            yield "status", data.get("status", "")
            yield "external_id", data.get("externalId")
            yield "created_at", data.get("createdAt", "")

        except Exception as e:
            yield "error", str(e)
            yield "webset_id", ""
            yield "status", ""
            yield "created_at", ""


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
        websets: list = SchemaField(description="List of websets", default_factory=list)
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
        expand_items: bool = SchemaField(
            default=False, description="Include items in the response", advanced=True
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

        params = {}
        if input_data.expand_items:
            params["expand[]"] = "items"

        try:
            response = await Requests().get(url, headers=headers, params=params)
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
