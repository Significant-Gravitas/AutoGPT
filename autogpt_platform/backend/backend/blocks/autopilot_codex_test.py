from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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
async def test_autopilot_block_rejects_codex_connection_change_on_resume():
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
    existing = SimpleNamespace(
        metadata=SimpleNamespace(
            llm_auth_provider="codex",
            llm_credential_id="different-credential",
        )
    )

    with (
        patch(
            "backend.copilot.model.get_chat_session",
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

    assert outputs == [
        ("session_id", "session-1"),
        ("error", "codex_session_route_mismatch"),
    ]
    execute_copilot.assert_not_awaited()
