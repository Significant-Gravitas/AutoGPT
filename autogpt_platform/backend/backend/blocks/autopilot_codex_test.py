from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.autopilot import AutoPilotBlock
from backend.data.execution import ExecutionContext
from backend.data.graph import NodeModel
from backend.integrations.providers import ProviderName


def test_codex_connection_is_a_top_level_reference_only_credential_field():
    schema = AutoPilotBlock.Input.jsonschema()
    field_schema = schema["properties"]["codex_credentials"]
    field_info = AutoPilotBlock.Input.get_credentials_fields_info()["codex_credentials"]

    assert "codex_credentials" in AutoPilotBlock.Input.get_credentials_fields()
    assert field_schema["credentials_provider"] == [ProviderName.CODEX]
    assert field_schema["credentials_types"] == ["oauth2"]
    assert field_schema["advanced"] is False
    assert field_schema["credential_reference_only"] is True
    assert field_info.provider == frozenset({ProviderName.CODEX})
    assert field_info.supported_types == frozenset({"oauth2"})
    assert field_info.credential_reference_only is True


def test_graph_export_strips_autopilot_codex_credential_reference():
    block = AutoPilotBlock()
    node = NodeModel(
        id="node-1",
        block_id=block.id,
        input_default={
            "prompt": "do the thing",
            "codex_credentials": {
                "id": "cred-1",
                "provider": "codex",
                "type": "oauth2",
                "title": "Personal ChatGPT",
            },
        },
        graph_id="graph-1",
        graph_version=1,
    )

    stripped = node.stripped_for_export()

    assert stripped.input_default == {"prompt": "do the thing"}


@pytest.mark.asyncio
async def test_autopilot_block_routes_new_session_to_selected_codex_connection():
    block = AutoPilotBlock()
    create_session = AsyncMock(return_value="session-1")
    execute_copilot = AsyncMock(
        return_value=(
            "done",
            [],
            "[]",
            "session-1",
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
    )
    input_data = AutoPilotBlock.Input(
        prompt="do the thing",
        codex_credentials={
            "id": "cred-1",
            "provider": "codex",
            "type": "oauth2",
            "title": "Personal ChatGPT",
        },
    )
    context = ExecutionContext(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="graph-exec-1",
        node_id="node-1",
        node_exec_id="node-exec-1",
    )

    with (
        patch.object(block, "create_session", create_session),
        patch.object(block, "execute_copilot", execute_copilot),
    ):
        outputs = {
            name: value
            async for name, value in block.run(
                input_data,
                execution_context=context,
            )
        }

    assert outputs["response"] == "done"
    assert outputs["session_id"] == "session-1"
    create_session.assert_awaited_once_with(
        "user-1",
        dry_run=False,
        organization_id=None,
        team_id=None,
        llm_auth_provider="codex",
        llm_credential_id="cred-1",
    )
    execute_copilot.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("existing", "error_fragment"),
    [
        (None, "session was not found"),
        (
            SimpleNamespace(
                metadata=SimpleNamespace(
                    llm_auth_provider="codex",
                    llm_credential_id="different-credential",
                )
            ),
            "different transport or connection",
        ),
    ],
)
async def test_autopilot_block_rejects_invalid_session_route(
    existing: SimpleNamespace | None,
    error_fragment: str,
):
    block = AutoPilotBlock()
    execute_copilot = AsyncMock()
    input_data = AutoPilotBlock.Input(
        prompt="continue",
        session_id="session-1",
        codex_credentials={
            "id": "cred-1",
            "provider": "codex",
            "type": "oauth2",
            "title": "Personal ChatGPT",
        },
    )
    context = ExecutionContext(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="graph-exec-1",
        node_id="node-1",
        node_exec_id="node-exec-1",
    )
    with (
        patch(
            "backend.copilot.model.get_chat_session_metadata",
            new=AsyncMock(return_value=existing),
        ),
        patch.object(block, "execute_copilot", execute_copilot),
    ):
        outputs = [
            item
            async for item in block.run(
                input_data,
                execution_context=context,
            )
        ]

    assert outputs[0] == ("session_id", "session-1")
    assert outputs[1][0] == "error"
    assert error_fragment in outputs[1][1]
    execute_copilot.assert_not_awaited()


def test_transport_is_an_explicit_choice_not_inferred_from_the_credential():
    """Which account pays used to be encoded as "is a credential set", visible
    only by noticing an empty field. `platform` spends AutoGPT credits;
    `codex_app_server` spends the user's own ChatGPT subscription."""

    schema = AutoPilotBlock.Input.jsonschema()
    transport = schema["properties"]["transport"]

    assert transport["advanced"] is False
    # Optional on purpose: a filled-in default cannot distinguish "never
    # chosen" from "deliberately platform", and reading one as the other
    # silently rebills legacy nodes.
    assert transport.get("default") is None
    assert transport["placeholder"] == "Select a transport"
    assert transport["anyOf"][0]["enumNames"] == ["AutoGPT Platform", "ChatGPT"]


def test_platform_transport_maps_to_no_provider():
    """`platform` is deliberately absent from the mapping: it needs no
    credential, and an unmapped value makes the credential input hide itself
    rather than asking for something that does not exist."""
    schema = AutoPilotBlock.Input.jsonschema()
    field_schema = schema["properties"]["codex_credentials"]

    assert field_schema["discriminator"] == "transport"
    assert field_schema["discriminator_mapping"] == {"codex_app_server": "codex"}
    assert "platform" not in field_schema["discriminator_mapping"]


def test_codex_credentials_stays_optional_under_the_discriminator():
    """Adding the discriminator must not make the connection mandatory —
    the platform transport is the default and needs nothing."""
    assert "codex_credentials" not in AutoPilotBlock.Input.get_required_fields()


def test_autopilot_does_not_use_discriminator_type_mapping():
    """Both of its options are a single credential type (or none), so the
    graph-aggregation fallback that CodeGeneration needs does not apply here.
    Pinned because `test_only_known_blocks_use_discriminator_type_mapping`
    asserts CodeGeneration is the sole user."""
    info = AutoPilotBlock.Input.get_credentials_fields_info()["codex_credentials"]

    assert not info.discriminator_type_mapping


def test_legacy_node_has_no_transport_so_its_connection_decides():
    """A node saved before the field existed carries a connection and no
    transport. That must stay distinguishable from a deliberate platform
    choice, or those agents get silently moved onto platform credits."""
    legacy = AutoPilotBlock.Input(
        prompt="do the thing",
        codex_credentials={
            "id": "codex-1",
            "provider": "codex",
            "type": "oauth2",
            "title": "ChatGPT for Codex",
        },
    )

    assert legacy.transport is None
    assert legacy.codex_credentials is not None


def test_explicit_platform_is_distinguishable_from_unset():
    """The whole reason the field is optional: a user moving a step back onto
    platform credits must not be mistaken for a legacy node."""
    from backend.blocks.autopilot import AutoPilotTransport

    chosen = AutoPilotBlock.Input(
        prompt="do the thing",
        transport=AutoPilotTransport.PLATFORM,
        codex_credentials={
            "id": "codex-1",
            "provider": "codex",
            "type": "oauth2",
            "title": "ChatGPT for Codex",
        },
    )

    assert chosen.transport == AutoPilotTransport.PLATFORM


CODEX_META = {
    "id": "cred-1",
    "provider": "codex",
    "type": "oauth2",
    "title": "Personal ChatGPT",
}


def _context():
    return ExecutionContext(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="graph-exec-1",
        node_id="node-1",
        node_exec_id="node-exec-1",
    )


async def _run_block(input_data):
    """Drive run() far enough to observe which account the session bills."""
    block = AutoPilotBlock()
    create_session = AsyncMock(return_value="session-1")
    execute_copilot = AsyncMock(
        return_value=(
            "done",
            [],
            "[]",
            "session-1",
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
    )
    with (
        patch.object(block, "create_session", create_session),
        patch.object(block, "execute_copilot", execute_copilot),
    ):
        outputs = {
            name: value
            async for name, value in block.run(input_data, execution_context=_context())
        }
    return outputs, create_session


@pytest.mark.asyncio
async def test_run_honours_explicit_platform_over_an_attached_connection():
    """Without this a user could never move a step back onto platform credits:
    the leftover connection kept deciding."""
    from backend.blocks.autopilot import AutoPilotTransport

    outputs, create_session = await _run_block(
        AutoPilotBlock.Input(
            prompt="go",
            transport=AutoPilotTransport.PLATFORM,
            codex_credentials=CODEX_META,
        )
    )

    assert outputs["response"] == "done"
    assert create_session.await_args.kwargs["llm_auth_provider"] == "platform"
    assert create_session.await_args.kwargs["llm_credential_id"] is None


@pytest.mark.asyncio
async def test_run_fails_closed_when_codex_is_chosen_without_a_connection():
    """Falling back to platform would bill a different account than the one
    asked for, without saying so."""
    from backend.blocks.autopilot import AutoPilotTransport

    outputs, create_session = await _run_block(
        AutoPilotBlock.Input(prompt="go", transport=AutoPilotTransport.CODEX_APP_SERVER)
    )

    assert "ChatGPT" in outputs["error"]
    create_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_resuming_a_codex_session_on_platform():
    """Regression: gating the resume check on `use_codex` skipped it entirely
    for a platform node, so it silently continued on the codex session and
    billed the subscription the user had just opted out of."""
    from backend.blocks.autopilot import AutoPilotTransport

    block = AutoPilotBlock()
    execute_copilot = AsyncMock()
    session = MagicMock()
    session.metadata.llm_auth_provider = "codex"
    session.metadata.llm_credential_id = "cred-1"

    input_data = AutoPilotBlock.Input(
        prompt="continue",
        session_id="session-1",
        transport=AutoPilotTransport.PLATFORM,
        codex_credentials=CODEX_META,
    )

    with (
        patch.object(block, "execute_copilot", execute_copilot),
        patch(
            "backend.copilot.model.get_chat_session_metadata",
            new=AsyncMock(return_value=session),
        ),
    ):
        outputs = {
            name: value
            async for name, value in block.run(input_data, execution_context=_context())
        }

    assert "different transport or connection" in outputs["error"]
    execute_copilot.assert_not_awaited()
