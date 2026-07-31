from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.codex import (
    TEST_CREDENTIALS_INPUT,
    CodexCallResult,
    CodeGenerationBlock,
    CodexExecutionTransport,
    CodexModel,
    CodexReasoningEffort,
    _app_server_effort,
)
from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.integrations.codex.models import (
    CodexInvocationResult,
    CodexTokenUsage,
)


def test_existing_graphs_default_to_openai_api_transport():
    input_data = CodeGenerationBlock.Input(
        prompt="preserve the old graph",
        credentials=TEST_CREDENTIALS_INPUT,
    )

    assert input_data.transport == CodexExecutionTransport.OPENAI_API
    field = CodeGenerationBlock.Input.model_fields["credentials"]
    assert field.json_schema_extra == {
        "discriminator": "transport",
        "discriminator_mapping": {
            "openai_api": "openai",
            "codex_app_server": "codex",
        },
        "discriminator_type_mapping": {
            "openai_api": ["api_key"],
            "codex_app_server": ["oauth2"],
        },
    }
    field_info = CodeGenerationBlock.Input.get_credentials_fields_info()["credentials"]
    assert field_info.discriminate("openai_api").supported_types == {"api_key"}
    assert field_info.discriminate("codex_app_server").supported_types == {"oauth2"}


def test_transport_selector_is_visible_before_transport_specific_fields():
    properties = CodeGenerationBlock.Input.jsonschema()["properties"]

    assert properties["transport"]["advanced"] is False
    assert list(properties).index("transport") < list(properties).index("model")
    assert list(properties).index("transport") < list(properties).index("credentials")


def test_subscription_none_effort_omits_model_specific_reasoning_value():
    assert _app_server_effort(CodexReasoningEffort.NONE) is None
    assert _app_server_effort(CodexReasoningEffort.HIGH) == "high"


@pytest.mark.asyncio
async def test_openai_api_transport_preserves_existing_call_path():
    block = CodeGenerationBlock()
    block.call_codex = AsyncMock(
        return_value=CodexCallResult(
            response="implemented",
            reasoning="used Responses API",
            response_id="response-1",
        )
    )
    credentials = APIKeyCredentials(
        id="openai-cred-1",
        provider="openai",
        api_key=SecretStr("secret"),
    )
    input_data = CodeGenerationBlock.Input(
        prompt="preserve it",
        credentials={
            "id": credentials.id,
            "provider": credentials.provider,
            "type": credentials.type,
            "title": credentials.title,
        },
    )

    outputs = [item async for item in block.run(input_data, credentials=credentials)]

    assert outputs == [
        ("response", "implemented"),
        ("reasoning", "used Responses API"),
        ("response_id", "response-1"),
    ]
    block.call_codex.assert_awaited_once_with(
        credentials=credentials,
        model=input_data.model,
        prompt="preserve it",
        system_prompt=input_data.system_prompt,
        max_output_tokens=input_data.max_output_tokens,
        reasoning_effort=input_data.reasoning_effort,
    )


@pytest.mark.asyncio
async def test_codex_app_server_uses_existing_lease_and_records_non_usd_usage():
    block = CodeGenerationBlock()
    credentials = _codex_credentials()
    lease = MagicMock(credentials=credentials)
    transport = MagicMock()
    transport.invoke = AsyncMock(
        return_value=CodexInvocationResult(
            response_id="turn-1",
            final_response="implemented",
            reasoning_summary="used the transport",
            status="completed",
            usage=CodexTokenUsage(
                input_tokens=100,
                cached_input_tokens=25,
                output_tokens=40,
                reasoning_output_tokens=10,
                total_tokens=140,
            ),
        )
    )
    input_data = CodeGenerationBlock.Input(
        prompt="implement it",
        system_prompt="be concise",
        model=CodexModel.GPT5_3_CODEX,
        reasoning_effort=CodexReasoningEffort.HIGH,
        transport=CodexExecutionTransport.CODEX_APP_SERVER,
        credentials=_credential_metadata(credentials),
    )

    with patch("backend.blocks.codex.get_codex_transport", return_value=transport):
        with patch(
            "backend.blocks.codex.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ):
            outputs = [
                item
                async for item in block.run(
                    input_data,
                    credentials=credentials,
                    credential_leases={"credentials": lease},
                    user_id="user-1",
                )
            ]

    assert outputs == [
        ("response", "implemented"),
        ("reasoning", "used the transport"),
        ("response_id", "turn-1"),
    ]
    invoke = transport.invoke.await_args.kwargs
    assert invoke["lease"] is lease
    assert invoke["request"].prompt == "implement it"
    assert invoke["request"].instructions == "be concise"
    assert invoke["request"].model is None
    assert invoke["request"].effort == "high"
    assert block.execution_stats.input_token_count == 100
    assert block.execution_stats.cache_read_token_count == 25
    assert block.execution_stats.output_token_count == 40
    assert block.execution_stats.provider_cost == 140
    assert block.execution_stats.provider_cost_type == "tokens"
    assert block.execution_stats.billing_mode == "user_subscription"
    assert block.execution_stats.execution_path == "codex_app_server"
    assert block.execution_stats.resolved_model is None


@pytest.mark.asyncio
async def test_codex_app_server_rejects_execution_without_credential_lease():
    block = CodeGenerationBlock()
    credentials = _codex_credentials()
    input_data = CodeGenerationBlock.Input(
        prompt="implement it",
        transport=CodexExecutionTransport.CODEX_APP_SERVER,
        credentials=_credential_metadata(credentials),
    )

    with (
        patch(
            "backend.blocks.codex.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
        pytest.raises(ValueError, match="credential lease"),
    ):
        await anext(block.run(input_data, credentials=credentials, user_id="user-1"))


@pytest.mark.asyncio
async def test_codex_app_server_requires_native_feature_flag():
    block = CodeGenerationBlock()
    credentials = _codex_credentials()
    lease = MagicMock(credentials=credentials)
    input_data = CodeGenerationBlock.Input(
        prompt="implement it",
        transport=CodexExecutionTransport.CODEX_APP_SERVER,
        credentials=_credential_metadata(credentials),
    )

    with (
        patch(
            "backend.blocks.codex.is_feature_enabled",
            new=AsyncMock(return_value=False),
        ) as flag_check,
        pytest.raises(ValueError, match="not enabled"),
    ):
        await anext(
            block.run(
                input_data,
                credentials=credentials,
                credential_leases={"credentials": lease},
                user_id="user-1",
            )
        )

    flag_check.assert_awaited_once_with(
        "codex-subscription-native",
        "user-1",
        default=False,
    )


def _codex_credentials() -> OAuth2Credentials:
    return OAuth2Credentials(
        id="codex-cred-1",
        provider="codex",
        access_token=SecretStr("access"),
        refresh_token=SecretStr("refresh"),
        scopes=[],
        refresh_strategy="provider_runtime",
        provider_state=SecretStr("{}"),
        provider_state_version=1,
    )


def _credential_metadata(credentials: OAuth2Credentials) -> dict[str, str | None]:
    return {
        "id": credentials.id,
        "provider": credentials.provider,
        "type": credentials.type,
        "title": credentials.title,
    }
