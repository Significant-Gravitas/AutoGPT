from typing import Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.dataforb2b._api import DataForB2BClient
from backend.blocks.dataforb2b._config import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    DataForB2BCredentials,
    DataForB2BCredentialsInput,
    dataforb2b,
)
from backend.blocks.dataforb2b._enums import CompanyColumn, FilterOperator, PeopleColumn
from backend.blocks.dataforb2b._filters import build_slot_condition, finalize_filters
from backend.data.model import SchemaField

NUM_SLOTS = 5

# Conservative safety ceiling for `count`. DataForB2B's documented API limits
# are not verifiable from this repo, so this bounds memory/response size
# without blocking legitimate use — callers needing more results should
# paginate with `offset` instead of raising this cap.
MAX_COUNT = 100

TEXT_OPERATORS = {
    FilterOperator.EQUALS,
    FilterOperator.NOT_EQUALS,
    FilterOperator.LIKE,
    FilterOperator.NOT_LIKE,
    FilterOperator.IN,
    FilterOperator.NOT_IN,
}
SCALAR_OPERATORS = {
    FilterOperator.EQUALS,
    FilterOperator.NOT_EQUALS,
    FilterOperator.IN,
    FilterOperator.NOT_IN,
    FilterOperator.GREATER_THAN,
    FilterOperator.GREATER_OR_EQUAL,
    FilterOperator.LESS_THAN,
    FilterOperator.LESS_OR_EQUAL,
    FilterOperator.BETWEEN,
}
BOOLEAN_OPERATORS = {
    FilterOperator.EQUALS,
    FilterOperator.NOT_EQUALS,
    FilterOperator.IN,
    FilterOperator.NOT_IN,
}
NUMERIC_COLUMNS = {
    PeopleColumn.follower_count,
    PeopleColumn.current_company_size,
    PeopleColumn.years_in_current_position,
    PeopleColumn.years_at_current_company,
    PeopleColumn.past_company_size,
    PeopleColumn.years_at_past_company,
    PeopleColumn.years_of_experience,
    PeopleColumn.num_total_jobs,
    CompanyColumn.employee_count,
    CompanyColumn.employee_growth_1m,
    CompanyColumn.employee_growth_6m,
    CompanyColumn.employee_growth_12m,
    CompanyColumn.recent_hires_count,
    CompanyColumn.founded_year,
    CompanyColumn.follower_count,
    CompanyColumn.last_funding_amount_usd,
    CompanyColumn.last_funding_date,
}
BOOLEAN_COLUMNS = {
    PeopleColumn.current_company_has_funding,
    PeopleColumn.is_currently_employed,
    CompanyColumn.page_verified,
    CompanyColumn.has_funding,
}


def _validate_operator(column, operator) -> None:
    if not column:
        return

    normalized_operator = operator or FilterOperator.EQUALS
    if column in BOOLEAN_COLUMNS:
        allowed = BOOLEAN_OPERATORS
    elif column in NUMERIC_COLUMNS:
        allowed = SCALAR_OPERATORS
    else:
        allowed = TEXT_OPERATORS

    if normalized_operator not in allowed:
        raise ValueError(
            f"Operator '{normalized_operator.value}' is not valid for column "
            f"'{column.value}'. Allowed operators: "
            f"{', '.join(sorted(op.value for op in allowed))}."
        )


def _build_filters(input_data) -> dict:
    conditions: list[dict] = []
    for i in range(1, NUM_SLOTS + 1):
        column = getattr(input_data, f"filter_{i}_column")
        operator = getattr(input_data, f"filter_{i}_operator")
        value = getattr(input_data, f"filter_{i}_value")
        if column and value is not None and str(value).strip():
            _validate_operator(column, operator)
        cond = build_slot_condition(column, operator, value)
        if cond:
            conditions.append(cond)

    advanced = input_data.filters_json or None
    filters = finalize_filters(conditions, input_data.match, advanced)
    if not filters:
        raise ValueError(
            "Provide at least one filter slot (column + value) or filters_json."
        )
    return filters


def _bounded_count_offset(input_data) -> tuple[int, int]:
    """Clamp count/offset to sane, non-negative bounds before calling the API."""
    count = max(1, min(int(input_data.count), MAX_COUNT))
    offset = max(0, int(input_data.offset))
    return count, offset


class _CommonSearchFields:
    """Fields shared by both search blocks (besides the column dropdowns)."""

    filters_json: dict = SchemaField(
        description=(
            "Raw filter JSON {op, conditions:[{column,type,value,value2?}]}. "
            "Paste 'applied_filters' from Smart Search and set 'offset' "
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
        description=f"Number of results to return (1-{MAX_COUNT})",
        default=25,
        advanced=False,
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
        title="Filter 1 Column",
        description="Filter 1 column",
        default=None,
        advanced=False,
    )
    filter_1_operator: FilterOperator = SchemaField(
        title="Filter 1 Operator",
        description="Filter 1 operator",
        default=FilterOperator.LIKE,
        advanced=False,
    )
    filter_1_value: str = SchemaField(
        title="Filter 1 Value", description="Filter 1 value", default="", advanced=False
    )
    filter_2_column: Optional[PeopleColumn] = SchemaField(
        title="Filter 2 Column",
        description="Filter 2 column",
        default=None,
        advanced=False,
    )
    filter_2_operator: FilterOperator = SchemaField(
        title="Filter 2 Operator",
        description="Filter 2 operator",
        default=FilterOperator.LIKE,
        advanced=False,
    )
    filter_2_value: str = SchemaField(
        title="Filter 2 Value", description="Filter 2 value", default="", advanced=False
    )
    filter_3_column: Optional[PeopleColumn] = SchemaField(
        title="Filter 3 Column",
        description="Filter 3 column",
        default=None,
        advanced=True,
    )
    filter_3_operator: FilterOperator = SchemaField(
        title="Filter 3 Operator",
        description="Filter 3 operator",
        default=FilterOperator.LIKE,
        advanced=True,
    )
    filter_3_value: str = SchemaField(
        title="Filter 3 Value", description="Filter 3 value", default="", advanced=True
    )
    filter_4_column: Optional[PeopleColumn] = SchemaField(
        title="Filter 4 Column",
        description="Filter 4 column",
        default=None,
        advanced=True,
    )
    filter_4_operator: FilterOperator = SchemaField(
        title="Filter 4 Operator",
        description="Filter 4 operator",
        default=FilterOperator.LIKE,
        advanced=True,
    )
    filter_4_value: str = SchemaField(
        title="Filter 4 Value", description="Filter 4 value", default="", advanced=True
    )
    filter_5_column: Optional[PeopleColumn] = SchemaField(
        title="Filter 5 Column",
        description="Filter 5 column",
        default=None,
        advanced=True,
    )
    filter_5_operator: FilterOperator = SchemaField(
        title="Filter 5 Operator",
        description="Filter 5 operator",
        default=FilterOperator.LIKE,
        advanced=True,
    )
    filter_5_value: str = SchemaField(
        title="Filter 5 Value", description="Filter 5 value", default="", advanced=True
    )


class _CompanyFilterFields(_CommonSearchFields):
    """Filter slots for company search — column is a CompanyColumn dropdown."""

    filter_1_column: Optional[CompanyColumn] = SchemaField(
        title="Filter 1 Column",
        description="Filter 1 column",
        default=None,
        advanced=False,
    )
    filter_1_operator: FilterOperator = SchemaField(
        title="Filter 1 Operator",
        description="Filter 1 operator",
        default=FilterOperator.LIKE,
        advanced=False,
    )
    filter_1_value: str = SchemaField(
        title="Filter 1 Value", description="Filter 1 value", default="", advanced=False
    )
    filter_2_column: Optional[CompanyColumn] = SchemaField(
        title="Filter 2 Column",
        description="Filter 2 column",
        default=None,
        advanced=False,
    )
    filter_2_operator: FilterOperator = SchemaField(
        title="Filter 2 Operator",
        description="Filter 2 operator",
        default=FilterOperator.LIKE,
        advanced=False,
    )
    filter_2_value: str = SchemaField(
        title="Filter 2 Value", description="Filter 2 value", default="", advanced=False
    )
    filter_3_column: Optional[CompanyColumn] = SchemaField(
        title="Filter 3 Column",
        description="Filter 3 column",
        default=None,
        advanced=True,
    )
    filter_3_operator: FilterOperator = SchemaField(
        title="Filter 3 Operator",
        description="Filter 3 operator",
        default=FilterOperator.LIKE,
        advanced=True,
    )
    filter_3_value: str = SchemaField(
        title="Filter 3 Value", description="Filter 3 value", default="", advanced=True
    )
    filter_4_column: Optional[CompanyColumn] = SchemaField(
        title="Filter 4 Column",
        description="Filter 4 column",
        default=None,
        advanced=True,
    )
    filter_4_operator: FilterOperator = SchemaField(
        title="Filter 4 Operator",
        description="Filter 4 operator",
        default=FilterOperator.LIKE,
        advanced=True,
    )
    filter_4_value: str = SchemaField(
        title="Filter 4 Value", description="Filter 4 value", default="", advanced=True
    )
    filter_5_column: Optional[CompanyColumn] = SchemaField(
        title="Filter 5 Column",
        description="Filter 5 column",
        default=None,
        advanced=True,
    )
    filter_5_operator: FilterOperator = SchemaField(
        title="Filter 5 Operator",
        description="Filter 5 operator",
        default=FilterOperator.LIKE,
        advanced=True,
    )
    filter_5_value: str = SchemaField(
        title="Filter 5 Value", description="Filter 5 value", default="", advanced=True
    )


class PeopleSearchBlock(Block):
    """Search LinkedIn people / B2B leads by structured filters with DataForB2B."""

    class Input(BlockSchemaInput, _PeopleFilterFields):
        credentials: DataForB2BCredentialsInput = dataforb2b.credentials_field(
            description="DataForB2B API key"
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="Full search response (total, count, results)"
        )
        results: list = SchemaField(
            description="List of matching LinkedIn people / leads", default_factory=list
        )
        total: int = SchemaField(description="Total number of matches", default=0)

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
            input_schema=PeopleSearchBlock.Input,
            output_schema=PeopleSearchBlock.Output,
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
        count, offset = _bounded_count_offset(input_data)
        payload = {
            "filters": _build_filters(input_data),
            "count": count,
            "offset": offset,
            "enrich_live": bool(input_data.enrich_live),
        }
        data = await self.search_people(payload, credentials)
        yield "result", data
        yield "results", data.get("results", []) or []
        yield "total", data.get("total", 0)


class CompanySearchBlock(Block):
    """Search LinkedIn companies by structured filters with DataForB2B."""

    class Input(BlockSchemaInput, _CompanyFilterFields):
        credentials: DataForB2BCredentialsInput = dataforb2b.credentials_field(
            description="DataForB2B API key"
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="Full search response (total, count, results)"
        )
        results: list = SchemaField(
            description="List of matching companies", default_factory=list
        )
        total: int = SchemaField(description="Total number of matches", default=0)

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
            input_schema=CompanySearchBlock.Input,
            output_schema=CompanySearchBlock.Output,
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
        count, offset = _bounded_count_offset(input_data)
        payload = {
            "filters": _build_filters(input_data),
            "count": count,
            "offset": offset,
            "enrich_live": bool(input_data.enrich_live),
        }
        data = await self.search_companies(payload, credentials)
        yield "result", data
        yield "results", data.get("results", []) or []
        yield "total", data.get("total", 0)
