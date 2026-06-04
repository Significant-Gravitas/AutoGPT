"""Tests for SetupTriggerTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks._base import BlockType
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.setup_agent_webhook_trigger import (
    SetupAgentWebhookTriggerTool,
    TriggerCredentialsRequiredResponse,
    TriggerSetupResponse,
)
from backend.data.model import APIKeyCredentials, CredentialsFieldInfo
from backend.integrations.providers import ProviderName

from ._test_data import make_session

_USER = "test-user-trigger"
_PATH = "backend.copilot.tools.setup_agent_webhook_trigger"


def _patch_triggers_db(preset=None):
    """Patch the triggers_db() accessor; returns (patch_ctx, setup mock)."""
    mock_client = MagicMock()
    mock_client.setup_triggered_preset = AsyncMock(return_value=preset)
    return (
        patch(f"{_PATH}.triggers_db", return_value=mock_client),
        mock_client.setup_triggered_preset,
    )


@pytest.fixture
def tool():
    return SetupAgentWebhookTriggerTool()


@pytest.fixture
def session():
    return make_session(_USER)


def _make_graph(*, manual: bool, required_credentials: dict | None = None):
    """Build a fake graph with a webhook trigger node."""
    node = MagicMock()
    node.id = "trigger-node"
    node.block.block_type = BlockType.WEBHOOK_MANUAL if manual else BlockType.WEBHOOK

    graph = MagicMock()
    graph.id = "graph-1"
    graph.version = 1
    graph.name = "My Webhook Agent"
    graph.webhook_input_node = node
    graph.regular_credentials_inputs = required_credentials or {}
    return graph


def _github_field_info() -> CredentialsFieldInfo:
    return CredentialsFieldInfo(
        credentials_provider=frozenset({ProviderName.GITHUB}),
        credentials_types=frozenset({"api_key"}),
    )


def _github_credential(cred_id: str = "cred-gh-1") -> APIKeyCredentials:
    return APIKeyCredentials(
        id=cred_id,
        provider="github",
        api_key=SecretStr("secret"),
        title="My GitHub",
    )


def _patch_graph_resolution(graph):
    """Patch graph_db()/library_db()/get_or_create_library_agent for resolution."""
    mock_graph_db = MagicMock()
    mock_graph_db.get_graph = AsyncMock(return_value=graph)
    return (
        patch(f"{_PATH}.graph_db", return_value=mock_graph_db),
        patch(
            f"{_PATH}.get_or_create_library_agent",
            new=AsyncMock(return_value=MagicMock(id="lib-1")),
        ),
    )


@pytest.mark.asyncio
async def test_setup_trigger_no_auth(tool, session):
    result = await tool._execute(user_id=None, session=session, name="My Trigger")
    assert isinstance(result, ErrorResponse)
    assert result.error == "auth_required"


@pytest.mark.asyncio
async def test_setup_trigger_missing_name(tool, session):
    result = await tool._execute(user_id=_USER, session=session, graph_id="graph-1")
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_argument"


@pytest.mark.asyncio
async def test_setup_trigger_no_webhook_node(tool, session):
    graph = _make_graph(manual=True)
    graph.webhook_input_node = None
    graph_db_patch, lib_patch = _patch_graph_resolution(graph)

    with graph_db_patch, lib_patch:
        result = await tool._execute(
            user_id=_USER, session=session, name="My Trigger", graph_id="graph-1"
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "no_webhook_trigger"


@pytest.mark.asyncio
async def test_setup_trigger_manual_returns_webhook_url(tool, session):
    graph = _make_graph(manual=True)  # no credentials required
    preset = MagicMock()
    preset.id = "preset-1"
    preset.name = "My Trigger"
    preset.is_active = True
    preset.webhook.url = (
        "https://backend.agpt.co/api/integrations/generic_webhook"
        "/webhooks/wh-1/ingress"
    )
    preset.webhook.provider = "generic_webhook"

    graph_db_patch, lib_patch = _patch_graph_resolution(graph)
    db_patch, mock_setup = _patch_triggers_db(preset)
    with (
        graph_db_patch,
        lib_patch,
        db_patch,
        patch(f"{_PATH}.get_user_credentials", new=AsyncMock(return_value=[])),
    ):
        result = await tool._execute(
            user_id=_USER, session=session, name="My Trigger", graph_id="graph-1"
        )

    assert isinstance(result, TriggerSetupResponse)
    assert result.manual_setup_required is True
    assert result.webhook_url and "/ingress" in result.webhook_url
    assert "backend.agpt.co" in result.webhook_url
    assert result.preset_id == "preset-1"
    mock_setup.assert_awaited_once()


@pytest.mark.asyncio
async def test_setup_trigger_provider_requires_explicit_credential(tool, session):
    """A provider webhook with one matching credential must NOT auto-pick it."""
    graph = _make_graph(
        manual=False,
        required_credentials={
            "github_credentials": (_github_field_info(), set(), None)
        },
    )
    graph_db_patch, lib_patch = _patch_graph_resolution(graph)
    db_patch, mock_setup = _patch_triggers_db()

    with (
        graph_db_patch,
        lib_patch,
        db_patch,
        patch(
            f"{_PATH}.get_user_credentials",
            new=AsyncMock(return_value=[_github_credential()]),
        ),
    ):
        result = await tool._execute(
            user_id=_USER,
            session=session,
            name="My Trigger",
            graph_id="graph-1",
            # no `credentials` -> must ask, not auto-pick
        )

    assert isinstance(result, TriggerCredentialsRequiredResponse)
    assert len(result.required_credentials) == 1
    field = result.required_credentials[0]
    assert field.field_name == "github_credentials"
    assert [opt.id for opt in field.options] == ["cred-gh-1"]
    # Critically: no preset created, no credential auto-attached.
    mock_setup.assert_not_called()


@pytest.mark.asyncio
async def test_setup_trigger_provider_no_credentials_lists_no_options(tool, session):
    """Provider webhook, user has 0 matching credentials -> empty options."""
    graph = _make_graph(
        manual=False,
        required_credentials={
            "github_credentials": (_github_field_info(), set(), None)
        },
    )
    graph_db_patch, lib_patch = _patch_graph_resolution(graph)
    db_patch, mock_setup = _patch_triggers_db()

    with (
        graph_db_patch,
        lib_patch,
        db_patch,
        patch(f"{_PATH}.get_user_credentials", new=AsyncMock(return_value=[])),
    ):
        result = await tool._execute(
            user_id=_USER, session=session, name="My Trigger", graph_id="graph-1"
        )

    assert isinstance(result, TriggerCredentialsRequiredResponse)
    assert result.required_credentials[0].options == []
    mock_setup.assert_not_called()


@pytest.mark.asyncio
async def test_setup_trigger_provider_with_explicit_credential_proceeds(tool, session):
    """When the user explicitly picks a credential ID, the preset is created."""
    graph = _make_graph(
        manual=False,
        required_credentials={
            "github_credentials": (_github_field_info(), set(), None)
        },
    )
    preset = MagicMock()
    preset.id = "preset-2"
    preset.name = "My Trigger"
    preset.is_active = True
    preset.webhook.url = (
        "https://backend.agpt.co/api/integrations/github/webhooks/wh-2/ingress"
    )
    preset.webhook.provider = "github"

    graph_db_patch, lib_patch = _patch_graph_resolution(graph)
    db_patch, mock_setup = _patch_triggers_db(preset)
    with (
        graph_db_patch,
        lib_patch,
        db_patch,
        patch(
            f"{_PATH}.get_user_credentials",
            new=AsyncMock(return_value=[_github_credential()]),
        ),
    ):
        result = await tool._execute(
            user_id=_USER,
            session=session,
            name="My Trigger",
            graph_id="graph-1",
            credentials={"github_credentials": "cred-gh-1"},
        )

    assert isinstance(result, TriggerSetupResponse)
    assert result.manual_setup_required is False
    mock_setup.assert_awaited_once()
    # The explicitly chosen credential is passed to the preset setup.
    passed_creds = mock_setup.await_args.kwargs["agent_credentials"]
    assert passed_creds["github_credentials"].id == "cred-gh-1"
