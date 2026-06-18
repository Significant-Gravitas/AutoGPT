from typing import Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.dataforb2b._api import DataForB2BClient
from backend.blocks.dataforb2b._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    DataForB2BCredentials,
    DataForB2BCredentialsField,
    DataForB2BCredentialsInput,
)
from backend.blocks.dataforb2b._enums import CompanyColumn, FilterOperator, PeopleColumn
from backend.blocks.dataforb2b._filters import build_slot_condition, finalize_filters
from backend.data.model import SchemaField

NUM_SLOTS = 5


def _build_filters(input_data) -> dict:
    conditions: list[dict] = []
    for i in range(1, NUM_SLOTS + 1):
        cond = build_slot_condition(
            getattr(input_data, f"filter_{i}_column"),
            getattr(input_data, f"filter_{i}_operator"),
            getattr(input_data, f"filter_{i}_value"),
        )
        if cond:
            conditions.append(cond)

    advanced = input_data.filters_json or None
    filters = finalize_filters(conditions, input_data.match, advanced)
    if not filters:
        raise ValueError(
            "Provide at least one filter slot (column + value) or filters_json."
        )
    return filters


class _CommonSearchFields:
    """Fields shared by both search blocks (besides the column dropdowns)."""

    filters_json: dict = SchemaField(
        description=(
            "Raw filter JSON {op, conditions:[{column,type,value,value2?}]}. "
            "Paste 'applied_filters' from Linkedin Smart Search and set 'offset' "
            "to paginate beyond the first page. Used alone, or merged (AND) with "
            "the filter slots above."
        ),
        default_factory=dict,
        advanced=False,
    )
    match: str = SchemaField(
        description="Combine slot conditions with 'and' or 'or'",
        default="and",
        advanced=True,
    )
    count: int = SchemaField(
        description="Number of results to return", default=25, advanced=False
    )
    offset: int = SchemaField(
        description="Pagination offset — 0 for page 1, then 25, 50, … to page through results",
        default=0,
        advanced=False,
    )
    enrich_live: bool = SchemaField(
        description="Fetch fresh live data (uses more credits)",
        default=False,
        advanced=True,
    )


class _PeopleFilterFields(_CommonSearchFields):
    """Filter slots for people search — column is a PeopleColumn dropdown."""

    filter_1_column: Optional[PeopleColumn] = SchemaField(
        description="Filter 1 column", default=None, advanced=False
    )
    filter_1_operator: FilterOperator = SchemaField(
        description="Filter 1 operator", default=FilterOperator.LIKE, advanced=False
    )
    filter_1_value: str = SchemaField(
        description="Filter 1 value", default="", advanced=False
    )
    filter_2_column: Optional[PeopleColumn] = SchemaField(
        description="Filter 2 column", default=None, advanced=False
    )
    filter_2_operator: FilterOperator = SchemaField(
        description="Filter 2 operator", default=FilterOperator.LIKE, advanced=False
    )
    filter_2_value: str = SchemaField(
        description="Filter 2 value", default="", advanced=False
    )
    filter_3_column: Optional[PeopleColumn] = SchemaField(
        description="Filter 3 column", default=None, advanced=True
    )
    filter_3_operator: FilterOperator = SchemaField(
        description="Filter 3 operator", default=FilterOperator.LIKE, advanced=True
    )
    filter_3_value: str = SchemaField(
        description="Filter 3 value", default="", advanced=True
    )
    filter_4_column: Optional[PeopleColumn] = SchemaField(
        description="Filter 4 column", default=None, advanced=True
    )
    filter_4_operator: FilterOperator = SchemaField(
        description="Filter 4 operator", default=FilterOperator.LIKE, advanced=True
    )
    filter_4_value: str = SchemaField(
        description="Filter 4 value", default="", advanced=True
    )
    filter_5_column: Optional[PeopleColumn] = SchemaField(
        description="Filter 5 column", default=None, advanced=True
    )
    filter_5_operator: FilterOperator = SchemaField(
        description="Filter 5 operator", default=FilterOperator.LIKE, advanced=True
    )
    filter_5_value: str = SchemaField(
        description="Filter 5 value", default="", advanced=True
    )


class _CompanyFilterFields(_CommonSearchFields):
    """Filter slots for company search — column is a CompanyColumn dropdown."""

    filter_1_column: Optional[CompanyColumn] = SchemaField(
        description="Filter 1 column", default=None, advanced=False
    )
    filter_1_operator: FilterOperator = SchemaField(
        description="Filter 1 operator", default=FilterOperator.LIKE, advanced=False
    )
    filter_1_value: str = SchemaField(
        description="Filter 1 value", default="", advanced=False
    )
    filter_2_column: Optional[CompanyColumn] = SchemaField(
        description="Filter 2 column", default=None, advanced=False
    )
    filter_2_operator: FilterOperator = SchemaField(
        description="Filter 2 operator", default=FilterOperator.LIKE, advanced=False
    )
    filter_2_value: str = SchemaField(
        description="Filter 2 value", default="", advanced=False
    )
    filter_3_column: Optional[CompanyColumn] = SchemaField(
        description="Filter 3 column", default=None, advanced=True
    )
    filter_3_operator: FilterOperator = SchemaField(
        description="Filter 3 operator", default=FilterOperator.LIKE, advanced=True
    )
    filter_3_value: str = SchemaField(
        description="Filter 3 value", default="", advanced=True
    )
    filter_4_column: Optional[CompanyColumn] = SchemaField(
        description="Filter 4 column", default=None, advanced=True
    )
    filter_4_operator: FilterOperator = SchemaField(
        description="Filter 4 operator", default=FilterOperator.LIKE, advanced=True
    )
    filter_4_value: str = SchemaField(
        description="Filter 4 value", default="", advanced=True
    )
    filter_5_column: Optional[CompanyColumn] = SchemaField(
        description="Filter 5 column", default=None, advanced=True
    )
    filter_5_operator: FilterOperator = SchemaField(
        description="Filter 5 operator", default=FilterOperator.LIKE, advanced=True
    )
    filter_5_value: str = SchemaField(
        description="Filter 5 value", default="", advanced=True
    )


class LinkedinPeopleSearchBlock(Block):
    """Search LinkedIn people / B2B leads by structured filters with DataForB2B."""

    class Input(BlockSchemaInput, _PeopleFilterFields):
        credentials: DataForB2BCredentialsInput = DataForB2BCredentialsField()

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="Full search response (total, count, results)"
        )
        results: list = SchemaField(
            description="List of matching LinkedIn people / leads", default_factory=list
        )
        total: int = SchemaField(description="Total number of matches", default=0)
        error: str = SchemaField(
            description="Error message if the search failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="91ded371-fe9f-432b-b0bd-d3788af485f8",
            description=(
                "Search people and B2B leads by structured filters — job title, company, "
                "location, industry, seniority, skills — using DataForB2B's database. "
                "Find employees at a company, people by job title, who works where, "
                "decision-makers and key contacts (owners, founders, C-suite, VPs, "
                "directors), and build a prospect or lead list. Accepts LinkedIn URLs as "
                "identifiers. The lead-sourcing step of a prospecting or outreach workflow."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.SOCIAL, BlockCategory.CRM},
            input_schema=LinkedinPeopleSearchBlock.Input,
            output_schema=LinkedinPeopleSearchBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "filter_1_column": PeopleColumn.current_title,
                "filter_1_operator": FilterOperator.LIKE,
                "filter_1_value": "software engineer",
                "count": 1,
            },
            test_output=[
                ("result", {"total": 1, "count": 1, "results": [{"id": "1"}]}),
                ("results", [{"id": "1"}]),
                ("total", 1),
            ],
            test_mock={
                "search_people": lambda payload, credentials: {
                    "total": 1,
                    "count": 1,
                    "results": [{"id": "1"}],
                }
            },
        )

    @staticmethod
    async def search_people(payload: dict, credentials: DataForB2BCredentials) -> dict:
        client = DataForB2BClient(credentials)
        return await client.search_people(payload)

    async def run(
        self, input_data: Input, *, credentials: DataForB2BCredentials, **kwargs
    ) -> BlockOutput:
        payload = {
            "filters": _build_filters(input_data),
            "count": int(input_data.count or 25),
            "offset": int(input_data.offset or 0),
            "enrich_live": bool(input_data.enrich_live),
        }
        data = await self.search_people(payload, credentials)
        yield "result", data
        yield "results", data.get("results", []) or []
        yield "total", data.get("total", 0)


class LinkedinCompanySearchBlock(Block):
    """Search LinkedIn companies by structured filters with DataForB2B."""

    class Input(BlockSchemaInput, _CompanyFilterFields):
        credentials: DataForB2BCredentialsInput = DataForB2BCredentialsField()

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="Full search response (total, count, results)"
        )
        results: list = SchemaField(
            description="List of matching companies", default_factory=list
        )
        total: int = SchemaField(description="Total number of matches", default=0)
        error: str = SchemaField(
            description="Error message if the search failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="4041c618-b7b8-4b74-aa3d-5b7cb07d6b1d",
            description=(
                "Search companies and accounts by structured filters — industry, "
                "headcount/size, location, funding, keywords — using DataForB2B's "
                "database. Build target-account lists for B2B sales and account-based "
                "marketing. Accepts LinkedIn URLs as identifiers."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.SOCIAL, BlockCategory.CRM},
            input_schema=LinkedinCompanySearchBlock.Input,
            output_schema=LinkedinCompanySearchBlock.Output,
            test_credentials=TEST_CREDENTIALS,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "filter_1_column": CompanyColumn.industry,
                "filter_1_operator": FilterOperator.LIKE,
                "filter_1_value": "software",
                "count": 1,
            },
            test_output=[
                ("result", {"total": 1, "count": 1, "results": [{"id": "1"}]}),
                ("results", [{"id": "1"}]),
                ("total", 1),
            ],
            test_mock={
                "search_companies": lambda payload, credentials: {
                    "total": 1,
                    "count": 1,
                    "results": [{"id": "1"}],
                }
            },
        )

    @staticmethod
    async def search_companies(
        payload: dict, credentials: DataForB2BCredentials
    ) -> dict:
        client = DataForB2BClient(credentials)
        return await client.search_companies(payload)

    async def run(
        self, input_data: Input, *, credentials: DataForB2BCredentials, **kwargs
    ) -> BlockOutput:
        payload = {
            "filters": _build_filters(input_data),
            "count": int(input_data.count or 25),
            "offset": int(input_data.offset or 0),
            "enrich_live": bool(input_data.enrich_live),
        }
        data = await self.search_companies(payload, credentials)
        yield "result", data
        yield "results", data.get("results", []) or []
        yield "total", data.get("total", 0)
