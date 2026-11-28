from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.hubspot._auth import (
    COMPANY_SCOPES,
    CONTACT_SCOPES,
    DEAL_READ_SCOPES,
    DEAL_SCOPES,
    HubSpotCredentials,
    HubSpotCredentialsField,
    HubSpotCredentialsInput,
)
from backend.data.model import SchemaField
from backend.util.request import Requests

API_URL = "https://api.hubapi.com"
DEALS_URL = f"{API_URL}/crm/v3/objects/deals"
# What HubSpot's search returns for a deal by default, plus the owner.
DEFAULT_DEAL_PROPERTIES = [
    "dealname",
    "amount",
    "dealstage",
    "pipeline",
    "closedate",
    "hubspot_owner_id",
    "createdate",
    "hs_lastmodifieddate",
]


class HubSpotDealStage(BaseModel):
    pipeline_id: str
    pipeline_label: str
    stage_id: str
    stage_label: str
    display_order: int
    probability: float | None
    is_closed: bool | None


class HubSpotDealBlock(Block):
    class Input(BlockSchemaInput):
        credentials: HubSpotCredentialsInput = HubSpotCredentialsField(DEAL_SCOPES)
        operation: Literal["create", "update"] = SchemaField(
            description="Operation to perform (create, update)",
            default="create",
            advanced=False,
        )
        deal_data: dict = SchemaField(
            description="Deal properties by internal name, e.g. dealname, amount, "
            "dealstage, pipeline, closedate. Stages and pipelines are set by ID.",
            default_factory=dict,
            advanced=False,
        )
        deal_id: str = SchemaField(
            description="ID of the deal to update", default="", advanced=False
        )

    class Output(BlockSchemaOutput):
        deal: dict = SchemaField(description="The deal as HubSpot returns it")
        status: str = SchemaField(description="Operation status")

    def __init__(self):
        super().__init__(
            id="170582a4-278b-48ba-8a16-3857a081dabb",
            description="Creates or updates a HubSpot deal",
            categories={BlockCategory.CRM},
            input_schema=HubSpotDealBlock.Input,
            output_schema=HubSpotDealBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    async def run(
        self, input_data: Input, *, credentials: HubSpotCredentials, **kwargs
    ) -> BlockOutput:
        body = {"properties": input_data.deal_data}
        if input_data.operation == "create":
            response = await Requests().post(
                DEALS_URL, headers=_headers(credentials), json=body
            )
            yield "deal", response.json()
            yield "status", "created"
            return

        if not input_data.deal_id:
            raise ValueError("Set deal_id to the ID of the deal to update")
        response = await Requests().patch(
            _deal_url(input_data.deal_id), headers=_headers(credentials), json=body
        )
        yield "deal", response.json()
        yield "status", "updated"


class HubSpotGetDealBlock(Block):
    class Input(BlockSchemaInput):
        credentials: HubSpotCredentialsInput = HubSpotCredentialsField(DEAL_READ_SCOPES)
        deal_id: str = SchemaField(description="ID of the deal")
        properties: list[str] = SchemaField(
            description="Deal properties to return. HubSpot leaves out a name it "
            "does not know.",
            default_factory=lambda: list(DEFAULT_DEAL_PROPERTIES),
        )

    class Output(BlockSchemaOutput):
        deal: dict = SchemaField(description="The deal as HubSpot returns it")
        contact_ids: list[str] = SchemaField(
            description="IDs of the contacts on the deal"
        )
        company_ids: list[str] = SchemaField(
            description="IDs of the companies on the deal"
        )
        status: str = SchemaField(description="Operation status")

    def __init__(self):
        super().__init__(
            id="a02c16e1-2710-4c61-b8a3-30cad8839b75",
            description="Gets a HubSpot deal by ID, with its contacts and companies",
            categories={BlockCategory.CRM},
            input_schema=HubSpotGetDealBlock.Input,
            output_schema=HubSpotGetDealBlock.Output,
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: HubSpotCredentials, **kwargs
    ) -> BlockOutput:
        response = await Requests().get(
            _deal_url(input_data.deal_id),
            headers=_headers(credentials),
            params={
                "properties": ",".join(input_data.properties),
                "associations": "contacts,companies",
            },
        )
        deal = response.json()
        associations = deal.get("associations") or {}
        yield "deal", deal
        yield "contact_ids", _associated_ids(associations, "contacts")
        yield "company_ids", _associated_ids(associations, "companies")
        yield "status", "retrieved"


class HubSpotSearchDealsBlock(Block):
    class Input(BlockSchemaInput):
        credentials: HubSpotCredentialsInput = HubSpotCredentialsField(DEAL_READ_SCOPES)
        query: str = SchemaField(
            description="Text to find in the deal name, pipeline, stage, "
            "description or type",
            default="",
            advanced=False,
        )
        pipeline_id: str = SchemaField(
            description="Only deals in this pipeline", default="", advanced=False
        )
        stage_id: str = SchemaField(
            description="Only deals in this stage", default="", advanced=False
        )
        filters: list[dict] = SchemaField(
            description="More HubSpot filters, all of which must match, e.g. "
            '{"propertyName": "amount", "operator": "GT", "value": "1000"}',
            default_factory=list,
        )
        sort_by: str = SchemaField(
            description="Deal property to sort by, e.g. closedate", default=""
        )
        sort_direction: Literal["ASCENDING", "DESCENDING"] = SchemaField(
            description="Sort order when sort_by is set", default="DESCENDING"
        )
        properties: list[str] = SchemaField(
            description="Deal properties to return",
            default_factory=lambda: list(DEFAULT_DEAL_PROPERTIES),
        )
        limit: int = SchemaField(
            description="Deals per page, at most 200", default=100, ge=1, le=200
        )
        after: str = SchemaField(
            description="next_after from the previous page, to get the next one",
            default="",
        )

    class Output(BlockSchemaOutput):
        deals: list[dict] = SchemaField(
            description="Matching deals, as HubSpot returns them"
        )
        total: int = SchemaField(description="How many deals match in all")
        next_after: str = SchemaField(
            description="Cursor for the next page; empty on the last page"
        )
        status: str = SchemaField(description="Operation status")

    def __init__(self):
        super().__init__(
            id="34fc71d1-3f07-4d15-a35a-67663849e4eb",
            description="Searches HubSpot deals by text, pipeline, stage or any "
            "deal property",
            categories={BlockCategory.CRM},
            input_schema=HubSpotSearchDealsBlock.Input,
            output_schema=HubSpotSearchDealsBlock.Output,
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: HubSpotCredentials, **kwargs
    ) -> BlockOutput:
        response = await Requests().post(
            f"{DEALS_URL}/search",
            headers=_headers(credentials),
            json=_search_body(input_data),
        )
        result = response.json()
        yield "deals", result.get("results", [])
        yield "total", result.get("total", 0)
        yield "next_after", ((result.get("paging") or {}).get("next") or {}).get(
            "after", ""
        )
        yield "status", "retrieved"


class HubSpotAssociateDealBlock(Block):
    class Input(BlockSchemaInput):
        credentials: HubSpotCredentialsInput = HubSpotCredentialsField(
            DEAL_SCOPES | CONTACT_SCOPES | COMPANY_SCOPES
        )
        deal_id: str = SchemaField(description="ID of the deal")
        object_type: Literal["contact", "company"] = SchemaField(
            description="What to associate the deal with",
            default="contact",
            advanced=False,
        )
        object_id: str = SchemaField(description="ID of the contact or company")

    class Output(BlockSchemaOutput):
        association: dict = SchemaField(
            description="HubSpot's report of the association it made"
        )
        status: str = SchemaField(description="Operation status")

    def __init__(self):
        super().__init__(
            id="175eac91-7687-46ed-bb06-f2c34358f8c1",
            description="Associates a HubSpot deal with a contact or a company",
            categories={BlockCategory.CRM},
            input_schema=HubSpotAssociateDealBlock.Input,
            output_schema=HubSpotAssociateDealBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    async def run(
        self, input_data: Input, *, credentials: HubSpotCredentials, **kwargs
    ) -> BlockOutput:
        deal_id = quote(input_data.deal_id, safe="")
        object_id = quote(input_data.object_id, safe="")
        response = await Requests().put(
            f"{API_URL}/crm/v4/objects/deal/{deal_id}/associations/default/"
            f"{input_data.object_type}/{object_id}",
            headers=_headers(credentials),
        )
        result = response.json()
        if result.get("numErrors"):
            raise ValueError(
                f"HubSpot did not associate the deal: {result.get('errors')}"
            )
        yield "association", result
        yield "status", "associated"


class HubSpotDealPipelinesBlock(Block):
    class Input(BlockSchemaInput):
        credentials: HubSpotCredentialsInput = HubSpotCredentialsField(DEAL_READ_SCOPES)

    class Output(BlockSchemaOutput):
        pipelines: list[dict] = SchemaField(
            description="Each deal pipeline with its stages, as HubSpot returns them"
        )
        stages: list[HubSpotDealStage] = SchemaField(
            description="Every stage of every pipeline, one row each; probability "
            "runs from 0 to 1"
        )
        status: str = SchemaField(description="Operation status")

    def __init__(self):
        super().__init__(
            id="27f4ea39-6a2c-4795-8ee8-5ec9daf8d03c",
            description="Lists the HubSpot deal pipelines and their stages",
            categories={BlockCategory.CRM},
            input_schema=HubSpotDealPipelinesBlock.Input,
            output_schema=HubSpotDealPipelinesBlock.Output,
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: HubSpotCredentials, **kwargs
    ) -> BlockOutput:
        response = await Requests().get(
            f"{API_URL}/crm/v3/pipelines/deals", headers=_headers(credentials)
        )
        pipelines = response.json().get("results", [])
        yield "pipelines", pipelines
        yield "stages", [
            _stage(pipeline, stage)
            for pipeline in pipelines
            for stage in pipeline.get("stages", [])
        ]
        yield "status", "retrieved"


def _headers(credentials: HubSpotCredentials) -> dict[str, str]:
    return {
        "Authorization": credentials.auth_header(),
        "Content-Type": "application/json",
    }


def _deal_url(deal_id: str) -> str:
    # Quoted so an ID can never move the request off the deal's own path.
    return f"{DEALS_URL}/{quote(deal_id, safe='')}"


def _associated_ids(associations: dict[str, Any], object_type: str) -> list[str]:
    # HubSpot lists a record once per association type, so an ID can repeat.
    results = (associations.get(object_type) or {}).get("results", [])
    return list(dict.fromkeys(str(r["id"]) for r in results))


def _search_body(input_data: HubSpotSearchDealsBlock.Input) -> dict[str, Any]:
    filters = list(input_data.filters)
    if input_data.pipeline_id:
        filters.append(
            {
                "propertyName": "pipeline",
                "operator": "EQ",
                "value": input_data.pipeline_id,
            }
        )
    if input_data.stage_id:
        filters.append(
            {
                "propertyName": "dealstage",
                "operator": "EQ",
                "value": input_data.stage_id,
            }
        )
    body: dict[str, Any] = {
        "filterGroups": [{"filters": filters}] if filters else [],
        "properties": input_data.properties,
        "limit": input_data.limit,
    }
    if input_data.query:
        body["query"] = input_data.query
    if input_data.sort_by:
        body["sorts"] = [
            {"propertyName": input_data.sort_by, "direction": input_data.sort_direction}
        ]
    if input_data.after:
        body["after"] = input_data.after
    return body


def _stage(pipeline: dict[str, Any], stage: dict[str, Any]) -> HubSpotDealStage:
    # The model parses HubSpot's string metadata ("0.2", "true").
    metadata = stage.get("metadata") or {}
    return HubSpotDealStage(
        pipeline_id=str(pipeline["id"]),
        pipeline_label=pipeline.get("label", ""),
        stage_id=str(stage["id"]),
        stage_label=stage.get("label", ""),
        display_order=stage.get("displayOrder", 0),
        probability=metadata.get("probability"),
        is_closed=metadata.get("isClosed"),
    )
