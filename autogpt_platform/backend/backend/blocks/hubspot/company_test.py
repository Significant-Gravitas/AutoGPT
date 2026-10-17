from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.blocks.hubspot._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.hubspot.company import HubSpotCompanyBlock
from backend.data.execution import ExecutionContext


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "results, expected_company, expected_status",
    [
        (
            [{"id": "company-1", "properties": {"domain": "example.com"}}],
            {"id": "company-1", "properties": {"domain": "example.com"}},
            "retrieved",
        ),
        ([], {}, "company_not_found"),
    ],
)
async def test_get_company_yields_declared_outputs(
    results: list[dict], expected_company: dict, expected_status: str
):
    response = Mock()
    response.json.return_value = {"results": results}
    post = AsyncMock(return_value=response)
    block = HubSpotCompanyBlock()

    with patch("backend.blocks.hubspot.company.Requests.post", post):
        outputs = [
            output
            async for output in block.execute(
                {"credentials": TEST_CREDENTIALS_INPUT, "domain": "example.com"},
                credentials=TEST_CREDENTIALS,
                execution_context=ExecutionContext(),
            )
        ]

    assert outputs == [("company", expected_company), ("status", expected_status)]
    assert post.await_count == 1
    assert post.call_args.kwargs["json"]["filterGroups"] == [
        {
            "filters": [
                {"propertyName": "domain", "operator": "EQ", "value": "example.com"}
            ]
        }
    ]


@pytest.mark.asyncio
async def test_update_missing_company_reports_not_found_without_patching():
    response = Mock()
    response.json.return_value = {"results": []}
    block = HubSpotCompanyBlock()

    with (
        patch(
            "backend.blocks.hubspot.company.Requests.post",
            AsyncMock(return_value=response),
        ),
        patch(
            "backend.blocks.hubspot.company.Requests.patch", new_callable=AsyncMock
        ) as update,
    ):
        outputs = [
            output
            async for output in block.execute(
                {
                    "credentials": TEST_CREDENTIALS_INPUT,
                    "domain": "example.com",
                    "operation": "update",
                    "company_data": {"name": "New name"},
                },
                credentials=TEST_CREDENTIALS,
                execution_context=ExecutionContext(),
            )
        ]

    assert outputs == [("company", {}), ("status", "company_not_found")]
    update.assert_not_awaited()
