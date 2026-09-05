import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from backend.data.notifications import SubscriptionPlan, TrialUpdateData
from backend.notifications import trial_postmark as postmark


@pytest.fixture
def settings():
    value = SimpleNamespace(
        config=SimpleNamespace(
            billing_sender_email="billing@example.com",
            postmark_transactional_stream="outbound",
        ),
        secrets=SimpleNamespace(postmark_server_api_token="test-token"),
    )
    with patch.object(postmark, "Settings", return_value=value):
        yield value


@pytest.fixture
def payload():
    return TrialUpdateData(
        user_name="Sam",
        kind="started",
        ends_label="17 Sep 2026",
        onboarding_credit_amount=300,
        offer_version="outbox-v1",
        plan=SubscriptionPlan(
            name="Pro",
            cycle="monthly",
            cycle_noun="month",
            label="Pro",
            price_display="$20.00 / month",
        ),
    )


@pytest.mark.asyncio
async def test_send_attaches_durable_identity_and_requires_provider_acceptance(
    settings, payload
):
    async_client = httpx.AsyncClient

    def handle(request):
        assert request.url.path == "/email"
        assert request.headers["X-Postmark-Server-Token"] == "test-token"
        body = json.loads(request.content)
        assert body["Metadata"] == {"trial_notice_id": "notice-1"}
        assert body["To"] == "sam@example.com"
        assert body["From"] == "billing@example.com"
        assert body["HtmlBody"] and body["TextBody"]
        assert body["MessageStream"] == "outbound"
        return httpx.Response(200, json={"MessageID": "accepted-1", "ErrorCode": 0})

    with patch.object(
        postmark.httpx,
        "AsyncClient",
        side_effect=lambda **kwargs: async_client(
            **kwargs, transport=httpx.MockTransport(handle)
        ),
    ):
        assert (
            await postmark.TrialEmailSender().send(
                "notice-1", "sam@example.com", payload
            )
            == "accepted-1"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("matched_id", ["notice-1", "another-notice"])
async def test_search_filters_and_validates_durable_metadata(settings, matched_id):
    async_client = httpx.AsyncClient

    def handle(request):
        assert request.url.path == "/messages/outbound"
        assert request.url.params["metadata_trial_notice_id"] == "notice-1"
        assert request.url.params["messagestream"] == "outbound"
        return httpx.Response(
            200,
            json={
                "Messages": [
                    {
                        "MessageID": "accepted-1",
                        "Metadata": {"trial_notice_id": matched_id},
                    }
                ]
            },
        )

    with patch.object(
        postmark.httpx,
        "AsyncClient",
        side_effect=lambda **kwargs: async_client(
            **kwargs, transport=httpx.MockTransport(handle)
        ),
    ):
        if matched_id == "notice-1":
            assert (
                await postmark.TrialEmailSender().find_accepted("notice-1")
                == "accepted-1"
            )
        else:
            with pytest.raises(ValueError, match="another trial"):
                await postmark.TrialEmailSender().find_accepted("notice-1")


@pytest.mark.asyncio
async def test_provider_outage_does_not_look_like_no_prior_send(settings):
    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    with patch.object(
        postmark.httpx,
        "AsyncClient",
        side_effect=lambda **kwargs: async_client(**kwargs, transport=transport),
    ):
        with pytest.raises(httpx.HTTPStatusError):
            await postmark.TrialEmailSender().find_accepted("notice-1")


@pytest.mark.asyncio
async def test_missing_postmark_configuration_is_a_delivery_failure(settings, payload):
    settings.secrets.postmark_server_api_token = ""
    with pytest.raises(RuntimeError, match="not configured"):
        await postmark.TrialEmailSender().send("notice-1", "sam@example.com", payload)
