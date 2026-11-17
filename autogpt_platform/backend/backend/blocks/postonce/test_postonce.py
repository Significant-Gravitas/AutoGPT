"""Unit tests for the PostOnce client and blocks.

Complements the ``test_mock`` harness in each block (run by
``test_available_blocks``) with the client's URLs, headers and error handling,
and the request body Create Post builds.
"""

import json
import runpy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from backend.blocks.postonce._api import API_V1, PostOnceAPIError, PostOnceClient
from backend.blocks.postonce.accounts import SAMPLE_ACCOUNTS, PostOnceListAccountsBlock
from backend.blocks.postonce.analytics import (
    SAMPLE_ANALYTICS,
    PostOnceGetPostAnalyticsBlock,
)
from backend.blocks.postonce.create_post import PostOnceCreatePostBlock, build_payload
from backend.sdk import APIKeyCredentials, SecretStr
from backend.sdk.registry import AutoRegistry

CREDENTIALS = APIKeyCredentials(
    id="0b4f7c1e-8a2d-4e5f-9c3b-6d1a2e3f4b5c",
    provider="postonce",
    api_key=SecretStr("mock-postonce-api-key"),
    title="Mock PostOnce API key",
    expires_at=None,
)
CREDENTIALS_INPUT = {
    "provider": CREDENTIALS.provider,
    "id": CREDENTIALS.id,
    "type": CREDENTIALS.type,
    "title": CREDENTIALS.title,
}


class FakeResponse:
    def __init__(self, status: int, body: Any):
        self.status = status
        self.ok = status < 400
        self._body = body
        self.content = json.dumps(body).encode()

    def json(self) -> Any:
        return self._body

    def text(self) -> str:
        return self.content.decode()


def client_with(response: FakeResponse) -> tuple[PostOnceClient, mock.AsyncMock]:
    client = PostOnceClient(CREDENTIALS)
    request = mock.AsyncMock(return_value=response)
    client.reads.request = request
    client.writes.request = request
    return client, request


async def collect(block, input_data) -> dict[str, Any]:
    return {
        name: value
        async for name, value in block.run(input_data, credentials=CREDENTIALS)
    }


def test_provider_takes_user_api_keys_only(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("POSTONCE_API_KEY", "server-key-must-not-be-shared")
    monkeypatch.setattr(AutoRegistry, "_providers", {})
    monkeypatch.setattr(AutoRegistry, "_default_credentials", [])
    monkeypatch.setattr(AutoRegistry, "_api_key_mappings", {})

    runpy.run_path(str(Path(__file__).with_name("_config.py")))

    provider = AutoRegistry.get_provider("postonce")
    assert provider is not None
    assert provider.supported_auth_types == {"api_key"}
    assert AutoRegistry.get_all_credentials() == []


@pytest.mark.asyncio
async def test_list_accounts_uses_bearer_auth_and_unwraps_data():
    client, request = client_with(FakeResponse(200, {"data": SAMPLE_ACCOUNTS}))
    assert await client.list_accounts() == SAMPLE_ACCOUNTS
    assert request.call_args.args == ("GET", f"{API_V1}/accounts")
    assert client.reads.extra_headers == {
        "Authorization": f"Bearer {CREDENTIALS.api_key.get_secret_value()}"
    }


@pytest.mark.asyncio
async def test_create_post_sends_idempotency_key_once():
    client, request = client_with(FakeResponse(201, {"data": {"id": "p1"}}))
    assert await client.create_post({"targets": []}, "key-1") == {"id": "p1"}
    assert request.call_args.args == ("POST", f"{API_V1}/posts")
    assert request.call_args.kwargs["headers"] == {"Idempotency-Key": "key-1"}
    assert client.writes.retry_max_attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call, method, path",
    [
        ("get_post", "GET", "/posts/p1"),
        ("cancel_post", "POST", "/posts/p1/cancel"),
        ("get_post_analytics", "GET", "/posts/p1/analytics"),
    ],
)
async def test_post_routes(call: str, method: str, path: str):
    client, request = client_with(FakeResponse(200, {"data": {"id": "p1"}}))
    assert await getattr(client, call)("p1") == {"id": "p1"}
    assert request.call_args.args == (method, f"{API_V1}{path}")


@pytest.mark.asyncio
async def test_api_error_carries_code_message_and_hint():
    body = {
        "error": {
            "code": "invalid_target",
            "message": "Account acc_9 is not connected.",
            "hint": "Reconnect it in PostOnce.",
        }
    }
    client, _ = client_with(FakeResponse(400, body))
    with pytest.raises(PostOnceAPIError) as error:
        await client.create_post({"targets": [{"account_id": "acc_9"}]})
    assert error.value.status == 400
    assert str(error.value) == (
        "PostOnce returned HTTP 400 (invalid_target): Account acc_9 is not "
        "connected. Reconnect it in PostOnce."
    )


def test_payload_maps_targets_media_and_schedule():
    publish_at = datetime(2030, 4, 1, 16, 0, tzinfo=timezone(timedelta(hours=2)))
    payload = build_payload(
        PostOnceCreatePostBlock.Input.model_validate(
            {
                "credentials": CREDENTIALS_INPUT,
                "account_ids": ["acc_1", "acc_2"],
                "content": "Hello",
                "media_urls": ["https://cdn.example.com/a.png"],
                "publish_at": publish_at,
                "content_overrides": {"acc_2": "Hi X"},
                "platform_options": {"acc_1": {"title": "Launch"}},
                "external_id": "launch-001",
            }
        )
    )
    assert payload == {
        "content": "Hello",
        "targets": [
            {"account_id": "acc_1", "platform_options": {"title": "Launch"}},
            {"account_id": "acc_2", "content_override": "Hi X"},
        ],
        "media": [{"url": "https://cdn.example.com/a.png"}],
        "publish_at": "2030-04-01T14:00:00Z",
        "external_id": "launch-001",
    }


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"content_overrides": {"acc_9": "x"}}, "not in account_ids: acc_9"),
        ({"content": " "}, "content or at least one media URL"),
    ],
)
def test_payload_rejects_bad_input(fields: dict, message: str):
    input_data = PostOnceCreatePostBlock.Input.model_validate(
        {
            "credentials": CREDENTIALS_INPUT,
            "account_ids": ["acc_1"],
            "content": "Hello",
            **fields,
        }
    )
    with pytest.raises(ValueError, match=message):
        build_payload(input_data)


@pytest.mark.asyncio
async def test_list_accounts_block_filters_inactive_accounts():
    block = PostOnceListAccountsBlock()
    with mock.patch.object(
        block, "_list", mock.AsyncMock(return_value=SAMPLE_ACCOUNTS)
    ):
        outputs = await collect(
            block,
            PostOnceListAccountsBlock.Input.model_validate(
                {"credentials": CREDENTIALS_INPUT}
            ),
        )
    assert outputs["account_ids"] == ["acc_1", "acc_3"]


@pytest.mark.asyncio
async def test_analytics_block_treats_missing_metrics_as_zero():
    block = PostOnceGetPostAnalyticsBlock()
    with mock.patch.object(
        block, "_analytics", mock.AsyncMock(return_value=SAMPLE_ANALYTICS)
    ):
        outputs = await collect(
            block,
            PostOnceGetPostAnalyticsBlock.Input.model_validate(
                {"credentials": CREDENTIALS_INPUT, "post_id": "p1"}
            ),
        )
    assert (outputs["views"], outputs["shares"], outputs["saves"]) == (1500, 4, 0)
