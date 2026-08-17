"""Tests for PeopleSearchBlock / CompanySearchBlock: filter
validation and count/offset clamping, with mock assertions on the actual
payload sent to the API (not just the returned block outputs)."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.dataforb2b._config import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_META_INPUT,
)
from backend.blocks.dataforb2b._enums import CompanyColumn, FilterOperator, PeopleColumn
from backend.blocks.dataforb2b.search import (
    MAX_COUNT,
    CompanyFilterCondition,
    CompanySearchBlock,
    PeopleFilterCondition,
    PeopleSearchBlock,
    _build_filters,
)


def _build(input_data) -> dict:
    return _build_filters(input_data.filters, input_data.match, input_data.filters_json)


@pytest.mark.asyncio
async def test_build_filters_raises_without_filters_or_advanced_json():
    """Thread ED6T: no filters and no filters_json must raise a clear ValueError
    instead of sending an empty/invalid filter payload to the API. The default
    filter row is a UI affordance with an empty value, so it must not count."""
    block = PeopleSearchBlock()
    input_data = PeopleSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
    )
    with pytest.raises(ValueError, match="at least one filter"):
        _build(input_data)

    with pytest.raises(ValueError, match="at least one filter"):
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass


@pytest.mark.asyncio
async def test_people_search_sends_full_filters_payload_to_api():
    """Thread ED6P: assert on the actual outgoing payload (filters/count/offset/
    enrich_live), not just the mocked return value."""
    block = PeopleSearchBlock()
    input_data = PeopleSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        filters=[
            PeopleFilterCondition(
                column=PeopleColumn.current_title,
                operator=FilterOperator.LIKE,
                value="software engineer",
            )
        ],
        count=5,
        offset=10,
        enrich_live=True,
    )
    with patch.object(
        PeopleSearchBlock,
        "search_people",
        new=AsyncMock(return_value={"total": 1, "count": 1, "results": [{"id": "1"}]}),
    ) as mock_search:
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass

    payload = mock_search.await_args.args[0]
    assert payload == {
        "filters": {
            "op": "and",
            "conditions": [
                {
                    "column": "current_title",
                    "type": "like",
                    "value": "software engineer",
                }
            ],
        },
        "count": 5,
        "offset": 10,
        "enrich_live": True,
    }


@pytest.mark.asyncio
async def test_filters_beyond_the_old_five_slot_cap_are_all_sent():
    """The list input removed the fixed 5-slot ceiling; every populated filter
    must reach the API."""
    block = PeopleSearchBlock()
    input_data = PeopleSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        filters=[
            PeopleFilterCondition(
                column=PeopleColumn.current_title,
                operator=FilterOperator.LIKE,
                value=f"title-{i}",
            )
            for i in range(7)
        ],
    )
    with patch.object(
        PeopleSearchBlock,
        "search_people",
        new=AsyncMock(return_value={"total": 0, "count": 0, "results": []}),
    ) as mock_search:
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass
    assert len(mock_search.await_args.args[0]["filters"]["conditions"]) == 7


@pytest.mark.asyncio
async def test_people_search_count_clamped_to_max():
    """Thread ED5-/DxBr: count above MAX_COUNT must be clamped, not sent raw."""
    block = PeopleSearchBlock()
    input_data = PeopleSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        filters=[
            PeopleFilterCondition(
                column=PeopleColumn.current_title,
                operator=FilterOperator.LIKE,
                value="engineer",
            )
        ],
        count=99999,
    )
    with patch.object(
        PeopleSearchBlock,
        "search_people",
        new=AsyncMock(return_value={"total": 0, "count": 0, "results": []}),
    ) as mock_search:
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass
    assert mock_search.await_args.args[0]["count"] == MAX_COUNT


@pytest.mark.asyncio
async def test_people_search_offset_clamped_to_non_negative():
    block = PeopleSearchBlock()
    input_data = PeopleSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        filters=[
            PeopleFilterCondition(
                column=PeopleColumn.current_title,
                operator=FilterOperator.LIKE,
                value="engineer",
            )
        ],
        offset=-5,
    )
    with patch.object(
        PeopleSearchBlock,
        "search_people",
        new=AsyncMock(return_value={"total": 0, "count": 0, "results": []}),
    ) as mock_search:
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass
    assert mock_search.await_args.args[0]["offset"] == 0


@pytest.mark.asyncio
async def test_company_search_raises_without_filters():
    """Thread ED6T (also applies to company search): same negative-path guard."""
    block = CompanySearchBlock()
    input_data = CompanySearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
    )
    with pytest.raises(ValueError, match="at least one filter"):
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass


@pytest.mark.asyncio
async def test_company_search_sends_full_filters_payload_to_api():
    block = CompanySearchBlock()
    input_data = CompanySearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        filters=[
            CompanyFilterCondition(
                column=CompanyColumn.industry,
                operator=FilterOperator.LIKE,
                value="software development",
            )
        ],
        count=1,
    )
    with patch.object(
        CompanySearchBlock,
        "search_companies",
        new=AsyncMock(return_value={"total": 1, "count": 1, "results": [{"id": "1"}]}),
    ) as mock_search:
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass
    payload = mock_search.await_args.args[0]
    assert payload["filters"] == {
        "op": "and",
        "conditions": [
            {"column": "industry", "type": "like", "value": "software development"}
        ],
    }
    assert payload["count"] == 1
    assert payload["offset"] == 0
    assert payload["enrich_live"] is False


@pytest.mark.asyncio
async def test_filters_json_alone_satisfies_filter_requirement():
    block = PeopleSearchBlock()
    input_data = PeopleSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        filters_json={
            "op": "and",
            "conditions": [{"column": "x", "type": "=", "value": 1}],
        },
    )
    with patch.object(
        PeopleSearchBlock,
        "search_people",
        new=AsyncMock(return_value={"total": 0, "count": 0, "results": []}),
    ) as mock_search:
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass
    assert mock_search.await_args.args[0]["filters"] == {
        "op": "and",
        "conditions": [{"column": "x", "type": "=", "value": 1}],
    }


def test_empty_filter_value_does_not_validate_operator():
    """A filter row left blank is inert — it must not trip operator validation
    for the column that happens to be selected."""
    input_data = PeopleSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        filters=[
            PeopleFilterCondition(
                column=PeopleColumn.years_of_experience,
                operator=FilterOperator.LIKE,
                value="",
            )
        ],
        filters_json={
            "op": "and",
            "conditions": [{"column": "current_title", "type": "like", "value": "cto"}],
        },
    )

    assert _build(input_data) == input_data.filters_json


@pytest.mark.parametrize(
    ("column", "operator"),
    [
        (PeopleColumn.is_currently_employed, FilterOperator.LIKE),
        (PeopleColumn.years_of_experience, FilterOperator.LIKE),
        (PeopleColumn.current_title, FilterOperator.GREATER_THAN),
    ],
)
def test_people_search_rejects_incompatible_column_operators(column, operator):
    input_data = PeopleSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        filters=[
            PeopleFilterCondition(column=column, operator=operator, value="1"),
        ],
    )

    with pytest.raises(ValueError, match="not valid for column"):
        _build(input_data)


@pytest.mark.parametrize(
    ("column", "operator"),
    [
        (CompanyColumn.page_verified, FilterOperator.LIKE),
        (CompanyColumn.employee_count, FilterOperator.LIKE),
        (CompanyColumn.industry, FilterOperator.BETWEEN),
    ],
)
def test_company_search_rejects_incompatible_column_operators(column, operator):
    input_data = CompanySearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        filters=[
            CompanyFilterCondition(column=column, operator=operator, value="1"),
        ],
    )

    with pytest.raises(ValueError, match="not valid for column"):
        _build(input_data)


def test_search_accepts_operator_compatible_with_column():
    input_data = PeopleSearchBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        filters=[
            PeopleFilterCondition(
                column=PeopleColumn.years_of_experience,
                operator=FilterOperator.BETWEEN,
                value="3,7",
            )
        ],
    )

    assert _build(input_data)["conditions"] == [
        {
            "column": "years_of_experience",
            "type": "between",
            "value": 3,
            "value2": 7,
        }
    ]


def test_default_filter_row_is_a_populated_example():
    """The blocks ship one pre-filled filter row so the input shape is obvious
    in the builder; its value is blank so it stays inert until edited."""
    people = PeopleSearchBlock.Input(credentials=TEST_CREDENTIALS_META_INPUT)
    company = CompanySearchBlock.Input(credentials=TEST_CREDENTIALS_META_INPUT)
    assert len(people.filters) == 1
    assert people.filters[0].column == PeopleColumn.current_title
    assert people.filters[0].operator == FilterOperator.EQUALS
    assert people.filters[0].value == ""
    assert len(company.filters) == 1
    assert company.filters[0].column == CompanyColumn.industry
