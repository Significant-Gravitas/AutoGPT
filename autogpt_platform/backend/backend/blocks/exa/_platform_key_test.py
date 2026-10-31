"""Team-wide Exa operations refuse the shared platform key."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.exa import _exa_rest, _platform_key, agent_billing
from backend.blocks.exa._exa_rest import error_message, path_segment
from backend.blocks.exa._platform_key import (
    PLATFORM_KEY_ID,
    is_platform_exa_key,
    require_own_exa_key,
)
from backend.blocks.exa._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.exa._webhook import ExaWebhookManager
from backend.blocks.exa.agent_manage import ExaListAgentRunsBlock
from backend.blocks.exa.search_monitors_manage import ExaListSearchMonitorsBlock
from backend.blocks.exa.websets_webhooks import ExaCreateWebsetWebhookBlock
from backend.data.model import APIKeyCredentials
from backend.integrations.credentials_store import exa_credentials

PLATFORM_KEY = APIKeyCredentials(
    id=PLATFORM_KEY_ID,
    provider="exa",
    api_key=SecretStr("platform-key"),
    title="Exa (platform)",
    expires_at=None,
)


def test_platform_keys_are_recognised():
    assert is_platform_exa_key(PLATFORM_KEY)
    assert is_platform_exa_key(exa_credentials)
    assert not is_platform_exa_key(TEST_CREDENTIALS)


def test_require_own_key_names_the_feature():
    require_own_exa_key(TEST_CREDENTIALS, "Exa Search Monitors")
    with pytest.raises(ValueError, match="Exa Search Monitors needs your own"):
        require_own_exa_key(PLATFORM_KEY, "Exa Search Monitors")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "block_cls, raw_input",
    [
        (ExaListSearchMonitorsBlock, {}),
        (
            ExaCreateWebsetWebhookBlock,
            {"url": "https://example.com/hook", "events": ["webset.created"]},
        ),
    ],
)
async def test_team_scoped_blocks_refuse_platform_key_before_any_request(
    block_cls, raw_input
):
    block = block_cls()
    request = AsyncMock()
    input_data = block.input_schema(credentials=TEST_CREDENTIALS_INPUT, **raw_input)
    with (
        patch.object(_platform_key, "exa_request", request),
        pytest.raises(ValueError, match="needs your own Exa API key"),
    ):
        async for _ in block.run(input_data, credentials=PLATFORM_KEY):
            pass
    request.assert_not_called()


@pytest.mark.asyncio
async def test_list_agent_runs_refuses_platform_key():
    block = ExaListAgentRunsBlock()
    input_data = block.input_schema(credentials=TEST_CREDENTIALS_INPUT)
    with (
        patch.object(block, "list_runs", AsyncMock()) as list_runs,
        pytest.raises(ValueError, match="Listing Exa Agent runs"),
    ):
        async for _ in block.run(input_data, credentials=PLATFORM_KEY):
            pass
    list_runs.assert_not_called()


@pytest.mark.asyncio
async def test_webset_trigger_refuses_to_register_on_platform_key():
    request = AsyncMock()
    with (
        patch.object(_exa_rest, "Requests", request),
        pytest.raises(ValueError, match="Exa Webset trigger"),
    ):
        await ExaWebhookManager()._register_webhook(
            PLATFORM_KEY,
            "webset",
            "",
            ["webset_created"],
            "https://platform.example/ingress",
            "secret",
        )
    request.assert_not_called()


def test_error_message_reads_nested_agent_api_errors():
    body = '{"error": {"type": "invalid_request", "code": "BAD_EFFORT", "message": "effort is invalid"}}'
    assert error_message(body) == "effort is invalid [BAD_EFFORT]"
    assert error_message('{"error": "Not found", "tag": "NOT_FOUND"}') == (
        "Not found [NOT_FOUND]"
    )


def test_path_segment_keeps_ids_inside_one_segment():
    assert path_segment("mon/../x?y") == "mon%2F..%2Fx%3Fy"


@pytest.mark.asyncio
async def test_billing_claim_bills_when_redis_is_unreachable():
    async def redis_down():
        raise ConnectionError("redis down")

    with patch.object(agent_billing, "get_redis_async", redis_down):
        assert await agent_billing.claim_run_billing("user-1", "agent_run_1")


@pytest.mark.asyncio
async def test_billing_claim_is_once_per_user_and_run(billing_claims):
    assert await agent_billing.claim_run_billing("user-1", "agent_run_1")
    assert not await agent_billing.claim_run_billing("user-1", "agent_run_1")
    assert await agent_billing.claim_run_billing("user-2", "agent_run_1")
