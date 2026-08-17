from typing import Optional

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
from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    SchemaField,
)

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


class PeopleFilterCondition(BlockSchemaInput):
    """One people-search filter: column + operator + value."""

    column: PeopleColumn = SchemaField(
        title="Column",
        description="Profile field to filter on",
        default=PeopleColumn.current_title,
        advanced=False,
    )
    operator: FilterOperator = SchemaField(
        title="Operator",
        description="How to compare. '=' is valid on every column; 'like' only on text",
        default=FilterOperator.EQUALS,
        advanced=False,
    )
    value: str = SchemaField(
        title="Value",
        description=(
            "Value to match. Search matches stored values exactly, so resolve it "
            "with Search Filter Typeahead first rather than guessing"
        ),
        default="",
        placeholder="e.g. Marketing Director",
        advanced=False,
    )


class CompanyFilterCondition(BlockSchemaInput):
    """One company-search filter: column + operator + value."""

    column: CompanyColumn = SchemaField(
        title="Column",
        description="Company field to filter on",
        default=CompanyColumn.industry,
        advanced=False,
    )
    operator: FilterOperator = SchemaField(
        title="Operator",
        description="How to compare. '=' is valid on every column; 'like' only on text",
        default=FilterOperator.EQUALS,
        advanced=False,
    )
    value: str = SchemaField(
        title="Value",
        description=(
            "Value to match. Search matches stored values exactly — 'software' finds "
            "nothing, 'software development' does. Resolve it with Search Filter "
            "Typeahead first"
        ),
        default="",
        placeholder="e.g. software development",
        advanced=False,
    )


def _validate_operator(
    column: Optional[PeopleColumn | CompanyColumn],
    operator: Optional[FilterOperator],
) -> None:
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


def _build_filters(
    conditions: list[PeopleFilterCondition] | list[CompanyFilterCondition],
    match: str,
    filters_json: dict,
) -> dict:
    built: list[dict] = []
    for cond in conditions:
        if cond.value is not None and str(cond.value).strip():
            _validate_operator(cond.column, cond.operator)
        slot = build_slot_condition(cond.column, cond.operator, cond.value)
        if slot:
            built.append(slot)

    filters = finalize_filters(built, match, filters_json or None)
    if not filters:
        raise ValueError(
            "Provide at least one filter (column + value) or filters_json."
        )
    return filters


def _bounded_count_offset(count: int, offset: int) -> tuple[int, int]:
    """Clamp count/offset to sane, non-negative bounds before calling the API."""
    return max(1, min(int(count), MAX_COUNT)), max(0, int(offset))


class PeopleSearchBlock(Block):
    """Search LinkedIn people / B2B leads by structured filters with DataForB2B."""

    class Input(BlockSchemaInput):
        filters: list[PeopleFilterCondition] = SchemaField(
            description=(
                "Filters to apply. Add one per field you want to narrow on; they are "
                "combined with 'match' (AND by default)."
            ),
            default_factory=lambda: [PeopleFilterCondition()],
            advanced=False,
        )
        count: int = SchemaField(
            description=f"Number of results to return (1-{MAX_COUNT})",
            default=25,
            advanced=False,
        )
        credentials: DataForB2BCredentialsInput = dataforb2b.credentials_field(
            description="DataForB2B API key"
        )
        offset: int = SchemaField(
            description="Pagination offset — 0 for page 1, then 25, 50, … to page through results",
            default=0,
            advanced=True,
        )
        match: str = SchemaField(
            description="Combine the filters above with 'and' or 'or'",
            default="and",
            advanced=True,
        )
        filters_json: dict = SchemaField(
            description=(
                "Escape hatch for filter shapes the list above cannot express, such as "
                "nested and/or groups. Paste 'applied_filters' from Smart Search here "
                "with an 'offset' to paginate its results. Merged (AND) with the "
                "filters above, or used alone."
            ),
            default_factory=dict,
            advanced=True,
        )
        enrich_live: bool = SchemaField(
            description="Fetch fresh live data (uses more credits)",
            default=False,
            advanced=True,
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
                "filters": [
                    {
                        "column": PeopleColumn.current_title,
                        "operator": FilterOperator.LIKE,
                        "value": "software engineer",
                    }
                ],
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
        count, offset = _bounded_count_offset(input_data.count, input_data.offset)
        payload = {
            "filters": _build_filters(
                input_data.filters, input_data.match, input_data.filters_json
            ),
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

    class Input(BlockSchemaInput):
        filters: list[CompanyFilterCondition] = SchemaField(
            description=(
                "Filters to apply. Add one per field you want to narrow on; they are "
                "combined with 'match' (AND by default)."
            ),
            default_factory=lambda: [CompanyFilterCondition()],
            advanced=False,
        )
        count: int = SchemaField(
            description=f"Number of results to return (1-{MAX_COUNT})",
            default=25,
            advanced=False,
        )
        credentials: DataForB2BCredentialsInput = dataforb2b.credentials_field(
            description="DataForB2B API key"
        )
        offset: int = SchemaField(
            description="Pagination offset — 0 for page 1, then 25, 50, … to page through results",
            default=0,
            advanced=True,
        )
        match: str = SchemaField(
            description="Combine the filters above with 'and' or 'or'",
            default="and",
            advanced=True,
        )
        filters_json: dict = SchemaField(
            description=(
                "Escape hatch for filter shapes the list above cannot express, such as "
                "nested and/or groups. Paste 'applied_filters' from Smart Search here "
                "with an 'offset' to paginate its results. Merged (AND) with the "
                "filters above, or used alone."
            ),
            default_factory=dict,
            advanced=True,
        )
        enrich_live: bool = SchemaField(
            description="Fetch fresh live data (uses more credits)",
            default=False,
            advanced=True,
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
                "filters": [
                    {
                        "column": CompanyColumn.industry,
                        "operator": FilterOperator.LIKE,
                        "value": "software development",
                    }
                ],
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
        count, offset = _bounded_count_offset(input_data.count, input_data.offset)
        payload = {
            "filters": _build_filters(
                input_data.filters, input_data.match, input_data.filters_json
            ),
            "count": count,
            "offset": offset,
            "enrich_live": bool(input_data.enrich_live),
        }
        data = await self.search_companies(payload, credentials)
        yield "result", data
        yield "results", data.get("results", []) or []
        yield "total", data.get("total", 0)
