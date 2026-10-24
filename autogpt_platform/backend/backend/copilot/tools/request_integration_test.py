"""request_integration: the "not supported yet" answer, and the request it records."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from backend.api.features.integrations import support
from backend.copilot.tools import request_integration
from backend.copilot.tools.models import (
    ErrorResponse,
    FeatureRequestCreatedResponse,
    IntegrationRequestResponse,
    ResponseType,
)
from backend.copilot.tools.request_integration import RequestIntegrationTool

from ._test_data import make_session

USER_ID = "test-user-request-integration"
REGISTRY = [
    support.ConnectableService(key="printify", name="Printify"),
    support.ConnectableService(key="etsy", name="Etsy"),
]


@pytest.fixture(autouse=True)
def registry():
    with patch.object(support, "connectable_services", return_value=REGISTRY):
        yield


def _filed(is_new: bool) -> FeatureRequestCreatedResponse:
    return FeatureRequestCreatedResponse(
        message="Filed.",
        issue_id="issue-1",
        issue_identifier="FR-1",
        issue_title="Integration request: Gelato",
        issue_url="https://linear.app/fr-1",
        is_new_issue=is_new,
        customer_name="someone",
    )


async def _run(**kwargs):
    return await RequestIntegrationTool()._execute(
        user_id=USER_ID, session=make_session(user_id=USER_ID), **kwargs
    )


async def test_an_unsupported_service_gets_the_card_and_joins_its_request(
    service_lock,
):
    with (
        patch.object(
            request_integration,
            "find_feature_request",
            AsyncMock(return_value="issue-1"),
        ) as find,
        patch.object(
            request_integration,
            "file_feature_request",
            AsyncMock(return_value=_filed(is_new=False)),
        ) as file,
    ):
        response = await _run(service=" Gelato ", use_case="Print sticker sheets")

    assert isinstance(response, IntegrationRequestResponse)
    assert response.type == ResponseType.INTEGRATION_REQUEST
    assert response.service == "Gelato"
    assert response.recorded is True
    find.assert_awaited_once_with("Integration request: Gelato")
    _, title, description, existing_issue_id, _ = file.await_args.args
    assert title == "Integration request: Gelato"
    assert description == "Print sticker sheets"
    assert existing_issue_id == "issue-1"
    # Otto is told not to fall back on a hand-built request unasked.
    assert "HTTP" in response.message


async def test_the_card_still_shows_when_the_request_cannot_be_recorded(
    service_lock,
):
    with (
        patch.object(
            request_integration, "find_feature_request", AsyncMock(return_value=None)
        ),
        patch.object(
            request_integration,
            "file_feature_request",
            AsyncMock(return_value=ErrorResponse(message="Linear is not configured")),
        ),
    ):
        response = await _run(service="Gelato", use_case="")

    assert isinstance(response, IntegrationRequestResponse)
    assert response.recorded is False


async def test_a_service_we_support_is_never_called_unsupported():
    with patch.object(request_integration, "file_feature_request", AsyncMock()) as file:
        response = await _run(service="printify", use_case="Make sticker products")

    assert isinstance(response, ErrorResponse)
    assert "find_capability" in response.message
    assert "Printify" in response.message
    file.assert_not_awaited()


async def test_a_blank_service_is_refused():
    response = await _run(service="  ", use_case="anything")
    assert isinstance(response, ErrorResponse)


@pytest.fixture
def service_lock():
    """A held lock stand-in, so filing never needs Redis in these tests."""
    lock = AsyncMock()
    with (
        patch.object(
            request_integration,
            "_acquire_service_lock",
            AsyncMock(return_value=lock),
        ) as acquire,
        patch.object(
            request_integration, "_release_service_lock", AsyncMock()
        ) as release,
    ):
        yield acquire, release, lock


async def test_a_failed_lookup_never_files_a_second_request(service_lock):
    """Not finding the request because Linear did not answer is not the same
    as there being none: filing then would split one service's asks."""
    with (
        patch.object(
            request_integration,
            "find_feature_request",
            AsyncMock(side_effect=RuntimeError("Linear timed out")),
        ),
        patch.object(request_integration, "file_feature_request", AsyncMock()) as file,
    ):
        response = await _run(service="Gelato", use_case="Print sticker sheets")

    assert isinstance(response, IntegrationRequestResponse)
    assert response.recorded is False
    file.assert_not_awaited()


async def test_one_tool_written_two_ways_joins_one_request(service_lock):
    with (
        patch.object(
            request_integration, "find_feature_request", AsyncMock(return_value=None)
        ) as find,
        patch.object(
            request_integration,
            "file_feature_request",
            AsyncMock(return_value=_filed(is_new=True)),
        ),
    ):
        response = await _run(service="Shopify store", use_case="")

    assert isinstance(response, IntegrationRequestResponse)
    assert response.service == "Shopify"
    find.assert_awaited_once_with("Integration request: Shopify")


async def test_one_service_is_looked_up_and_filed_under_its_lock(service_lock):
    acquire, release, lock = service_lock
    with (
        patch.object(
            request_integration, "find_feature_request", AsyncMock(return_value=None)
        ),
        patch.object(
            request_integration,
            "file_feature_request",
            AsyncMock(return_value=_filed(is_new=True)),
        ),
    ):
        response = await _run(service="Gelato", use_case="")

    assert isinstance(response, IntegrationRequestResponse)
    assert response.recorded is True
    acquire.assert_awaited_once_with("Gelato")
    release.assert_awaited_once_with(lock)


async def test_without_the_lock_nothing_is_filed_and_the_card_still_shows():
    with (
        patch.object(
            request_integration, "_acquire_service_lock", AsyncMock(return_value=None)
        ),
        patch.object(request_integration, "file_feature_request", AsyncMock()) as file,
    ):
        response = await _run(service="Gelato", use_case="")

    assert isinstance(response, IntegrationRequestResponse)
    assert response.recorded is False
    file.assert_not_awaited()


async def test_a_linear_outage_still_returns_the_card_in_time(service_lock):
    """However long Linear takes, the user gets the card, unrecorded, and
    the service's lock is given back."""
    _, release, lock = service_lock

    async def never_answers(title: str) -> str | None:
        await asyncio.Event().wait()
        return None

    with (
        patch.object(request_integration, "RECORD_TIMEOUT_SECONDS", 0.05),
        patch.object(request_integration, "find_feature_request", never_answers),
        patch.object(request_integration, "file_feature_request", AsyncMock()) as file,
    ):
        response = await asyncio.wait_for(
            _run(service="Gelato", use_case="Print sticker sheets"), timeout=5
        )

    assert isinstance(response, IntegrationRequestResponse)
    assert response.recorded is False
    file.assert_not_awaited()
    release.assert_awaited_once_with(lock)
