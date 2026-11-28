from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from pytest_mock import MockerFixture

from backend.blocks._base import Block
from backend.blocks.hubspot._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.hubspot.deal import (
    DEFAULT_DEAL_PROPERTIES,
    HubSpotAssociateDealBlock,
    HubSpotDealBlock,
    HubSpotDealPipelinesBlock,
    HubSpotDealStage,
    HubSpotGetDealBlock,
    HubSpotSearchDealsBlock,
)

DEALS = "https://api.hubapi.com/crm/v3/objects/deals"


@pytest.fixture
def hubspot(mocker: MockerFixture):
    """Answers one HTTP method of the patched ``Requests`` with ``body``."""
    requests = mocker.patch("backend.blocks.hubspot.deal.Requests").return_value

    def respond(method: str, body: dict[str, Any]) -> AsyncMock:
        response = Mock()
        response.json.return_value = body
        setattr(requests, method, AsyncMock(return_value=response))
        return getattr(requests, method)

    return respond


@pytest.mark.parametrize(
    ("inputs", "method", "url", "status"),
    [
        ({"operation": "create"}, "post", DEALS, "created"),
        ({"operation": "update", "deal_id": "123"}, "patch", f"{DEALS}/123", "updated"),
    ],
)
async def test_create_and_update_send_the_properties(
    hubspot, inputs, method, url, status
):
    request = hubspot(method, {"id": "123"})
    deal_data = {"dealname": "Acme renewal", "amount": "5000", "dealstage": "s1"}

    outputs = await _run(HubSpotDealBlock(), deal_data=deal_data, **inputs)

    sent_url, kwargs = _sent(request)
    assert sent_url == url
    assert kwargs["json"] == {"properties": deal_data}
    assert outputs == {"deal": {"id": "123"}, "status": status}


async def test_update_without_a_deal_id_sends_nothing(hubspot):
    request = hubspot("patch", {})

    with pytest.raises(ValueError, match="deal_id"):
        await _run(HubSpotDealBlock(), operation="update", deal_data={"amount": "1"})

    request.assert_not_awaited()


async def test_get_asks_for_associations_and_lists_each_record_once(hubspot):
    request = hubspot(
        "get",
        {
            "id": "123",
            "properties": {"dealname": "Acme renewal"},
            "associations": {
                "companies": {
                    "results": [
                        {"id": "7", "type": "deal_to_company"},
                        {"id": "7", "type": "deal_to_company_unlabeled"},
                    ]
                },
                "contacts": {"results": [{"id": "3", "type": "deal_to_contact"}]},
            },
        },
    )

    outputs = await _run(HubSpotGetDealBlock(), deal_id="123", properties=["amount"])

    sent_url, kwargs = _sent(request)
    assert sent_url == f"{DEALS}/123"
    assert kwargs["params"] == {
        "properties": "amount",
        "associations": "contacts,companies",
    }
    assert outputs["contact_ids"] == ["3"]
    assert outputs["company_ids"] == ["7"]
    assert outputs["status"] == "retrieved"


async def test_get_of_a_deal_with_no_associations_lists_none(hubspot):
    hubspot("get", {"id": "123", "properties": {}})

    outputs = await _run(HubSpotGetDealBlock(), deal_id="123")

    assert outputs["contact_ids"] == []
    assert outputs["company_ids"] == []


async def test_a_deal_id_cannot_leave_the_deal_path(hubspot):
    request = hubspot("get", {"id": "2"})

    await _run(HubSpotGetDealBlock(), deal_id="1/../../companies/2")

    sent_url, _ = _sent(request)
    assert sent_url == f"{DEALS}/1%2F..%2F..%2Fcompanies%2F2"


@pytest.mark.parametrize(
    ("inputs", "body", "response", "next_after"),
    [
        (
            {},
            {"filterGroups": [], "properties": DEFAULT_DEAL_PROPERTIES, "limit": 100},
            {"total": 1, "results": [{"id": "1"}]},
            "",
        ),
        (
            {
                "query": "acme",
                "pipeline_id": "default",
                "stage_id": "s1",
                "filters": [
                    {"propertyName": "amount", "operator": "GT", "value": "10"}
                ],
                "sort_by": "closedate",
                "sort_direction": "ASCENDING",
                "properties": ["dealname"],
                "limit": 2,
                "after": "2",
            },
            {
                "filterGroups": [
                    {
                        "filters": [
                            {"propertyName": "amount", "operator": "GT", "value": "10"},
                            {
                                "propertyName": "pipeline",
                                "operator": "EQ",
                                "value": "default",
                            },
                            {
                                "propertyName": "dealstage",
                                "operator": "EQ",
                                "value": "s1",
                            },
                        ]
                    }
                ],
                "properties": ["dealname"],
                "limit": 2,
                "query": "acme",
                "sorts": [{"propertyName": "closedate", "direction": "ASCENDING"}],
                "after": "2",
            },
            {
                "total": 5,
                "results": [{"id": "3"}, {"id": "4"}],
                "paging": {"next": {"after": "4"}},
            },
            "4",
        ),
    ],
    ids=["nothing set", "every input set"],
)
async def test_search_sends_one_group_of_filters_and_pages_on(
    hubspot, inputs, body, response, next_after
):
    request = hubspot("post", response)

    outputs = await _run(HubSpotSearchDealsBlock(), **inputs)

    sent_url, kwargs = _sent(request)
    assert sent_url == f"{DEALS}/search"
    assert kwargs["json"] == body
    assert outputs == {
        "deals": response["results"],
        "total": response["total"],
        "next_after": next_after,
        "status": "retrieved",
    }


@pytest.mark.parametrize("object_type", ["contact", "company"])
async def test_associate_creates_the_default_association(hubspot, object_type):
    result = {"status": "COMPLETE", "results": [{"from": {"id": "9"}}], "numErrors": 0}
    request = hubspot("put", result)

    outputs = await _run(
        HubSpotAssociateDealBlock(),
        deal_id="9",
        object_type=object_type,
        object_id="42",
    )

    sent_url, _ = _sent(request)
    assert sent_url == (
        "https://api.hubapi.com/crm/v4/objects/deal/9/associations/default/"
        f"{object_type}/42"
    )
    assert outputs == {"association": result, "status": "associated"}


async def test_associate_fails_when_hubspot_reports_an_error(hubspot):
    hubspot(
        "put", {"status": "COMPLETE", "numErrors": 1, "errors": [{"message": "no"}]}
    )

    with pytest.raises(ValueError, match="did not associate"):
        await _run(HubSpotAssociateDealBlock(), deal_id="9", object_id="42")


async def test_pipelines_are_listed_with_one_row_per_stage(hubspot):
    pipelines = [
        {
            "id": "default",
            "label": "Sales",
            "stages": [
                {
                    "id": "s1",
                    "label": "Qualified",
                    "displayOrder": 0,
                    "metadata": {"probability": "0.2", "isClosed": "false"},
                },
                {
                    "id": "s2",
                    "label": "Closed won",
                    "displayOrder": 1,
                    "metadata": {"probability": "1.0", "isClosed": "true"},
                },
            ],
        },
        {"id": "p2", "label": "Renewals", "stages": [{"id": "r1", "label": "Open"}]},
    ]
    request = hubspot("get", {"results": pipelines})

    outputs = await _run(HubSpotDealPipelinesBlock())

    sent_url, _ = _sent(request)
    assert sent_url == "https://api.hubapi.com/crm/v3/pipelines/deals"
    assert outputs["pipelines"] == pipelines
    assert outputs["stages"] == [
        HubSpotDealStage(
            pipeline_id="default",
            pipeline_label="Sales",
            stage_id="s1",
            stage_label="Qualified",
            display_order=0,
            probability=0.2,
            is_closed=False,
        ),
        HubSpotDealStage(
            pipeline_id="default",
            pipeline_label="Sales",
            stage_id="s2",
            stage_label="Closed won",
            display_order=1,
            probability=1.0,
            is_closed=True,
        ),
        HubSpotDealStage(
            pipeline_id="p2",
            pipeline_label="Renewals",
            stage_id="r1",
            stage_label="Open",
            display_order=0,
            probability=None,
            is_closed=None,
        ),
    ]


async def _run(block: Block, **inputs: Any) -> dict[str, Any]:
    input_data = block.input_schema(credentials=TEST_CREDENTIALS_INPUT, **inputs)
    return {
        name: value
        async for name, value in block.run(input_data, credentials=TEST_CREDENTIALS)
    }


def _sent(request: AsyncMock) -> tuple[str, dict[str, Any]]:
    """The one request a block sent, after checking it carried the token."""
    request.assert_awaited_once()
    assert request.await_args is not None
    url, *_ = request.await_args.args
    kwargs = request.await_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer mock-hubspot-access-token"
    return url, kwargs
