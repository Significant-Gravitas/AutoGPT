from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.copilot.rate_limit import UserPaywalledError
from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.model import (
    APIKeyCredentials,
    CredentialsFieldInfo,
    CredentialsMetaInput,
    NodeExecutionStats,
    OAuth2Credentials,
)
from backend.executor.manager import execute_node
from backend.integrations.providers import ProviderName

CodexCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.CODEX], Literal["oauth2"]
]
MixedCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.OPENAI, ProviderName.CODEX],
    Literal["api_key", "oauth2"],
]


@pytest.mark.asyncio
async def test_execute_node_injects_and_releases_credential_lease():
    credentials = _codex_credentials("cred-1")
    lease = MagicMock(credentials=credentials)
    lease.release = AsyncMock()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=lease)
    captured: dict[str, object] = {}
    block = _block_with_credentials({"credentials": CodexCredentialsInput}, captured)

    outputs = [
        item
        async for item in execute_node(
            _node(block),
            _entry({"credentials": _credential_metadata("cred-1")}),
            SimpleNamespace(creds_manager=manager),
        )
    ]

    assert outputs == [("response", "ok")]
    assert captured["credentials"] is credentials
    assert captured["credential_leases"] == {"credentials": lease}
    manager.acquire_lease.assert_awaited_once_with("user-1", "cred-1")
    lease.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_node_does_not_inject_runtime_lease_for_api_key():
    credentials = APIKeyCredentials(
        id="cred-1",
        provider="openai",
        api_key=SecretStr("secret"),
    )
    lease = MagicMock(credentials=credentials)
    lease.release = AsyncMock()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=lease)
    captured: dict[str, object] = {}
    openai_credentials = CredentialsMetaInput[
        Literal[ProviderName.OPENAI], Literal["api_key"]
    ]
    block = _block_with_credentials({"credentials": openai_credentials}, captured)

    outputs = [
        item
        async for item in execute_node(
            _node(block),
            _entry(
                {
                    "credentials": {
                        "id": "cred-1",
                        "provider": "openai",
                        "type": "api_key",
                        "title": None,
                    }
                }
            ),
            SimpleNamespace(creds_manager=manager),
        )
    ]

    assert outputs == [("response", "ok")]
    assert captured["credentials"] is credentials
    assert "credential_leases" not in captured
    lease.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_node_releases_first_lease_when_second_acquisition_fails():
    first = MagicMock(credentials=_codex_credentials("cred-1"))
    first.release = AsyncMock()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(
        side_effect=[first, RuntimeError("second acquisition failed")]
    )
    block = _block_with_credentials(
        {
            "first_credentials": CodexCredentialsInput,
            "second_credentials": CodexCredentialsInput,
        },
        {},
    )
    inputs = {
        "first_credentials": _credential_metadata("cred-1"),
        "second_credentials": _credential_metadata("cred-2"),
    }

    with pytest.raises(RuntimeError, match="second acquisition"):
        await anext(
            execute_node(
                _node(block),
                _entry(inputs),
                SimpleNamespace(creds_manager=manager),
            )
        )

    first.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_reference_only_credential_is_validated_without_outer_lease():
    credentials = _codex_credentials("cred-1")
    manager = MagicMock()
    manager.get = AsyncMock(return_value=credentials)
    manager.acquire_lease = AsyncMock()
    captured: dict[str, Any] = {}
    block = _block_with_credentials(
        {"codex_credentials": CodexCredentialsInput},
        captured,
        reference_only_fields={"codex_credentials"},
    )
    metadata = _credential_metadata("cred-1")

    outputs = [
        item
        async for item in execute_node(
            _node(block),
            _entry({"codex_credentials": metadata}),
            SimpleNamespace(creds_manager=manager),
        )
    ]

    assert outputs == [("response", "ok")]
    manager.get.assert_awaited_once_with("user-1", "cred-1")
    manager.acquire_lease.assert_not_awaited()
    assert captured["input_data"]["codex_credentials"] == metadata
    assert "codex_credentials" not in captured
    assert "credential_leases" not in captured


def _discriminated_reference_block(
    captured: dict[str, object],
    *,
    required: bool = False,
) -> MagicMock:
    block = _block_with_credentials(
        {"codex_credentials": CodexCredentialsInput},
        captured,
        reference_only_fields={"codex_credentials"},
    )
    block.input_schema.get_credentials_fields_info.return_value = {
        "codex_credentials": CredentialsFieldInfo(
            credentials_provider=frozenset({ProviderName.CODEX}),
            credentials_types=frozenset({"oauth2"}),
            credential_reference_only=True,
            discriminator="transport",
            discriminator_mapping={"codex_app_server": ProviderName.CODEX},
        )
    }
    block.input_schema.get_required_fields.return_value = (
        {"codex_credentials"} if required else set()
    )
    return block


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_in_defaults", [False, True])
async def test_platform_transport_ignores_attached_codex_credential(
    transport_in_defaults: bool,
):
    manager = MagicMock()
    manager.get = AsyncMock()
    manager.acquire_lease = AsyncMock()
    captured: dict[str, Any] = {}
    block = _discriminated_reference_block(captured)
    gate = AsyncMock()
    node = _node(block)
    inputs: dict[str, object] = {"codex_credentials": _credential_metadata("cred-1")}
    if transport_in_defaults:
        node.input_default = {"transport": "platform"}
    else:
        inputs["transport"] = "platform"

    validation = MagicMock(side_effect=lambda _node, inputs, **_kwargs: (inputs, None))
    with (
        patch("backend.executor.manager.enforce_codex_access", new=gate),
        patch("backend.executor.manager.validate_exec", new=validation),
    ):
        outputs = [
            item
            async for item in execute_node(
                node,
                _entry(inputs),
                SimpleNamespace(creds_manager=manager),
            )
        ]

    assert outputs == [("response", "ok")]
    assert validation.call_args.args[1]["codex_credentials"] is None
    manager.get.assert_not_awaited()
    manager.acquire_lease.assert_not_awaited()
    gate.assert_not_awaited()
    assert captured["input_data"]["codex_credentials"] is None
    assert "codex_credentials" not in captured


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", [None, "codex_app_server"])
async def test_legacy_and_codex_transports_resolve_attached_codex_credential(
    transport: str | None,
):
    credentials = _codex_credentials("cred-1")
    manager = MagicMock()
    manager.get = AsyncMock(return_value=credentials)
    manager.acquire_lease = AsyncMock()
    captured: dict[str, Any] = {}
    block = _discriminated_reference_block(captured)
    gate = AsyncMock()
    inputs: dict[str, object] = {"codex_credentials": _credential_metadata("cred-1")}
    if transport is not None:
        inputs["transport"] = transport

    with patch("backend.executor.manager.enforce_codex_access", new=gate):
        outputs = [
            item
            async for item in execute_node(
                _node(block),
                _entry(inputs),
                SimpleNamespace(creds_manager=manager),
            )
        ]

    assert outputs == [("response", "ok")]
    manager.get.assert_awaited_once_with("user-1", "cred-1")
    gate.assert_awaited_once_with("user-1")
    assert captured["input_data"]["codex_credentials"]["id"] == "cred-1"


@pytest.mark.asyncio
async def test_required_unmapped_credential_still_fails_closed():
    manager = MagicMock()
    manager.get = AsyncMock(return_value=None)
    manager.acquire_lease = AsyncMock()
    block = _discriminated_reference_block({}, required=True)

    with pytest.raises(ValueError, match="Credentials #cred-1 .* not found"):
        await anext(
            execute_node(
                _node(block),
                _entry(
                    {
                        "transport": "platform",
                        "codex_credentials": _credential_metadata("cred-1"),
                    }
                ),
                SimpleNamespace(creds_manager=manager),
            )
        )

    manager.get.assert_awaited_once_with("user-1", "cred-1")


@pytest.mark.asyncio
async def test_codex_credential_is_rejected_and_released_without_entitlement():
    credentials = _codex_credentials("cred-1")
    lease = MagicMock(credentials=credentials)
    lease.release = AsyncMock()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=lease)
    block = _block_with_credentials({"credentials": CodexCredentialsInput}, {})
    gate = AsyncMock(side_effect=UserPaywalledError("Max plan required"))

    with (
        patch("backend.executor.manager.enforce_codex_access", new=gate),
        pytest.raises(UserPaywalledError, match="Max plan required"),
    ):
        await anext(
            execute_node(
                _node(block),
                _entry({"credentials": _credential_metadata("cred-1")}),
                SimpleNamespace(creds_manager=manager),
            )
        )

    gate.assert_awaited_once_with("user-1")
    manager.acquire_lease.assert_awaited_once_with("user-1", "cred-1")
    lease.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_credential_metadata_cannot_hide_authoritative_codex_provider():
    credentials = _codex_credentials("cred-1")
    lease = MagicMock(credentials=credentials)
    lease.release = AsyncMock()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=lease)
    captured: dict[str, object] = {}
    block = _block_with_credentials({"credentials": MixedCredentialsInput}, captured)
    gate = AsyncMock()
    disguised_metadata = {
        "id": "cred-1",
        "provider": "openai",
        "type": "oauth2",
        "title": None,
    }

    with (
        patch("backend.executor.manager.enforce_codex_access", new=gate),
        pytest.raises(ValueError, match="Credentials #cred-1 .* not found"),
    ):
        await anext(
            execute_node(
                _node(block),
                _entry({"credentials": disguised_metadata}),
                SimpleNamespace(creds_manager=manager),
            )
        )

    gate.assert_not_awaited()
    lease.release.assert_awaited_once()
    assert "credentials" not in captured


def _block_with_credentials(
    credential_fields: dict[str, type[CodexCredentialsInput]],
    captured: dict[str, object],
    *,
    reference_only_fields: set[str] | None = None,
) -> MagicMock:
    reference_only_fields = reference_only_fields or set()
    input_schema = MagicMock()
    input_schema.get_credentials_fields.return_value = credential_fields
    # A real `CredentialsFieldInfo`, not a stand-in carrying only the one
    # attribute the test happens to need: `execute_node` reads whatever the
    # field declares, so a partial fake fails with an AttributeError the moment
    # the executor consults another one.
    input_schema.get_credentials_fields_info.return_value = {
        field_name: CredentialsFieldInfo(
            credentials_provider=frozenset({ProviderName.CODEX}),
            credentials_types=frozenset({"oauth2"}),
            credential_reference_only=field_name in reference_only_fields,
        )
        for field_name in credential_fields
    }
    input_schema.get_auto_credentials_fields.return_value = {}
    block = MagicMock()
    block.id = "block-1"
    block.name = "LeaseBlock"
    block.disabled = False
    block.input_schema = input_schema
    block.execution_stats = NodeExecutionStats()

    async def execute(_input_data, **kwargs):
        captured["input_data"] = _input_data
        captured.update(kwargs)
        yield "response", "ok"

    block.execute = execute
    return block


def _node(block: MagicMock) -> MagicMock:
    node = MagicMock()
    node.block = block
    node.input_default = {}
    return node


def _entry(inputs: dict[str, object]) -> NodeExecutionEntry:
    return NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id="graph-exec-1",
        graph_id="graph-1",
        graph_version=1,
        node_exec_id="node-exec-1",
        node_id="node-1",
        block_id="block-1",
        inputs=inputs,
        execution_context=ExecutionContext(),
    )


def _credential_metadata(credentials_id: str) -> dict[str, str | None]:
    return {
        "id": credentials_id,
        "provider": "codex",
        "type": "oauth2",
        "title": None,
    }


def _codex_credentials(credentials_id: str) -> OAuth2Credentials:
    return OAuth2Credentials(
        id=credentials_id,
        provider="codex",
        access_token=SecretStr("access"),
        refresh_token=SecretStr("refresh"),
        scopes=[],
        refresh_strategy="provider_runtime",
        provider_state=SecretStr("{}"),
        provider_state_version=1,
    )


@pytest.fixture(autouse=True)
def _execution_boundaries():
    scope = MagicMock(_user=None, _tags={})
    with (
        patch(
            "backend.executor.manager.validate_exec",
            side_effect=lambda _node, data, **_kwargs: (data, None),
        ),
        patch("backend.executor.manager._sentry_get_current_scope", return_value=scope),
    ):
        yield
