"""Tests for SetupAgentWebhookTriggerTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks._base import BlockType
from backend.copilot.tools.models import ErrorResponse, SetupRequirementsResponse
from backend.copilot.tools.setup_agent_webhook_trigger import (
    SetupAgentWebhookTriggerTool,
    TriggerConfigRequiredResponse,
    TriggerSetupResponse,
)
from backend.data.model import (
    APIKeyCredentials,
    CredentialsFieldInfo,
    CredentialsMetaInput,
)
from backend.integrations.providers import ProviderName
from backend.util.exceptions import InvalidInputError

from ._test_data import make_session

_USER = "test-user-trigger"
_PATH = "backend.copilot.tools.setup_agent_webhook_trigger"


@pytest.fixture
def tool():
    return SetupAgentWebhookTriggerTool()


@pytest.fixture
def session():
    return make_session(_USER)


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


def _github_meta(cred_id: str = "cred-gh-1") -> CredentialsMetaInput:
    return CredentialsMetaInput(
        id=cred_id,
        provider=ProviderName.GITHUB,
        type="api_key",
        title="My GitHub",
    )


def _make_graph(
    *,
    manual: bool,
    regular_credentials: dict | None = None,
    config_schema: dict | None = None,
):
    """Build a fake graph with a webhook trigger node.

    ``config_schema`` defaults to an empty (no-required) schema so the trigger
    config check passes; pass one with ``required`` to exercise that path.
    """
    node = MagicMock()
    node.id = "trigger-node"
    node.block.block_type = BlockType.WEBHOOK_MANUAL if manual else BlockType.WEBHOOK

    graph = MagicMock()
    graph.id = "graph-1"
    graph.version = 1
    graph.name = "My Webhook Agent"
    graph.webhook_input_node = node
    graph.regular_credentials_inputs = regular_credentials or {}
    graph.trigger_setup_info = MagicMock(config_schema=config_schema or {})
    return graph


def _make_preset(*, provider: str, url: str):
    preset = MagicMock()
    preset.id = "preset-1"
    preset.name = "My Trigger"
    preset.is_active = True
    preset.webhook.url = url
    preset.webhook.provider = provider
    return preset


def _patches(graph, *, matched=None, missing=None, available=None, preset=None):
    """Patch graph resolution + credential matching + DB calls for the tool."""
    mock_graph_db = MagicMock()
    mock_graph_db.get_graph = AsyncMock(return_value=graph)
    mock_triggers = MagicMock()
    mock_triggers.setup_triggered_preset = AsyncMock(return_value=preset)
    return [
        patch(f"{_PATH}.graph_db", return_value=mock_graph_db),
        patch(
            f"{_PATH}.get_or_create_library_agent",
            new=AsyncMock(return_value=MagicMock(id="lib-1")),
        ),
        patch(
            f"{_PATH}.match_user_credentials_to_graph",
            new=AsyncMock(return_value=(matched or {}, missing or [])),
        ),
        patch(
            f"{_PATH}.get_user_credentials",
            new=AsyncMock(return_value=available or []),
        ),
        patch(f"{_PATH}.triggers_db", return_value=mock_triggers),
    ], mock_triggers.setup_triggered_preset


@pytest.mark.asyncio
async def test_no_auth(tool, session):
    result = await tool._execute(user_id=None, session=session, name="My Trigger")
    assert isinstance(result, ErrorResponse)
    assert result.error == "auth_required"


@pytest.mark.asyncio
async def test_missing_name(tool, session):
    result = await tool._execute(user_id=_USER, session=session, graph_id="graph-1")
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_argument"


@pytest.mark.asyncio
async def test_no_webhook_node(tool, session):
    graph = _make_graph(manual=True)
    graph.webhook_input_node = None
    ctxs, _ = _patches(graph)
    with ctxs[0], ctxs[1], ctxs[2], ctxs[3], ctxs[4]:
        result = await tool._execute(
            user_id=_USER, session=session, name="My Trigger", graph_id="graph-1"
        )
    assert isinstance(result, ErrorResponse)
    assert result.error == "no_webhook_trigger"


@pytest.mark.asyncio
async def test_manual_webhook_no_creds_proceeds(tool, session):
    """Manual webhook with no required credentials: create the preset, return URL."""
    graph = _make_graph(manual=True, regular_credentials={})
    preset = _make_preset(
        provider="generic_webhook",
        url="https://backend.agpt.co/api/integrations/generic_webhook/webhooks/wh-1/ingress",
    )
    ctxs, setup_mock = _patches(graph, preset=preset)
    with ctxs[0], ctxs[1], ctxs[2], ctxs[3], ctxs[4]:
        result = await tool._execute(
            user_id=_USER, session=session, name="My Trigger", graph_id="graph-1"
        )
    assert isinstance(result, TriggerSetupResponse)
    assert result.manual_setup_required is True
    assert result.webhook_url and "backend.agpt.co" in result.webhook_url
    setup_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_manual_webhook_missing_body_cred_returns_card(tool, session):
    """Manual webhook whose body needs a (missing) credential -> connect card."""
    graph = _make_graph(
        manual=True,
        regular_credentials={
            "e2b_credentials": (
                _github_field_info(),
                {("body-node", "credentials")},
                True,
            )
        },
    )
    # No match -> the body cred is missing.
    ctxs, setup_mock = _patches(graph, matched={}, missing=["e2b_credentials"])
    with ctxs[0], ctxs[1], ctxs[2], ctxs[3], ctxs[4]:
        result = await tool._execute(
            user_id=_USER, session=session, name="My Trigger", graph_id="graph-1"
        )
    assert isinstance(result, SetupRequirementsResponse)
    assert "e2b_credentials" in result.setup_info.user_readiness.missing_credentials
    setup_mock.assert_not_called()


@pytest.mark.asyncio
async def test_provider_webhook_first_call_surfaces_trigger_cred(tool, session):
    """Provider webhook: the trigger account is shown in the card even when a
    candidate auto-matched, and the preset is NOT created yet."""
    graph = _make_graph(
        manual=False,
        regular_credentials={
            "github_credentials": (
                _github_field_info(),
                {("trigger-node", "credentials")},
                True,
            )
        },
    )
    # Auto-match would pick the single account, but it must still be surfaced.
    ctxs, setup_mock = _patches(graph, matched={"github_credentials": _github_meta()})
    with ctxs[0], ctxs[1], ctxs[2], ctxs[3], ctxs[4]:
        result = await tool._execute(
            user_id=_USER, session=session, name="My Trigger", graph_id="graph-1"
        )
    assert isinstance(result, SetupRequirementsResponse)
    assert "github_credentials" in result.setup_info.user_readiness.missing_credentials
    setup_mock.assert_not_called()


@pytest.mark.asyncio
async def test_provider_webhook_explicit_selection_proceeds(tool, session):
    """Provider webhook resumed with an explicit account -> preset created with it."""
    graph = _make_graph(
        manual=False,
        regular_credentials={
            "github_credentials": (
                _github_field_info(),
                {("trigger-node", "credentials")},
                True,
            )
        },
    )
    preset = _make_preset(
        provider="github",
        url="https://backend.agpt.co/api/integrations/github/webhooks/wh-2/ingress",
    )
    ctxs, setup_mock = _patches(
        graph,
        matched={"github_credentials": _github_meta()},
        available=[_github_credential()],
        preset=preset,
    )
    with ctxs[0], ctxs[1], ctxs[2], ctxs[3], ctxs[4]:
        result = await tool._execute(
            user_id=_USER,
            session=session,
            name="My Trigger",
            graph_id="graph-1",
            credentials={"github_credentials": "cred-gh-1"},
        )
    assert isinstance(result, TriggerSetupResponse)
    assert result.manual_setup_required is False
    setup_mock.assert_awaited_once()
    passed = setup_mock.await_args.kwargs["agent_credentials"]
    assert passed["github_credentials"].id == "cred-gh-1"


@pytest.mark.asyncio
async def test_provider_webhook_missing_config_asks_user(tool, session):
    """Required trigger config (repo/events) the LLM didn't supply -> ask the
    user conversationally; don't attempt setup or guess values."""
    graph = _make_graph(
        manual=False,
        regular_credentials={
            "github_credentials": (
                _github_field_info(),
                {("trigger-node", "credentials")},
                True,
            )
        },
        config_schema={
            "properties": {"repo": {"type": "string"}, "events": {"type": "object"}},
            "required": ["repo", "events"],
        },
    )
    ctxs, setup_mock = _patches(graph)
    with ctxs[0], ctxs[1], ctxs[2], ctxs[3], ctxs[4]:
        result = await tool._execute(
            user_id=_USER, session=session, name="My Trigger", graph_id="graph-1"
        )

    assert isinstance(result, TriggerConfigRequiredResponse)
    assert set(result.missing_config) == {"repo", "events"}
    setup_mock.assert_not_called()


@pytest.mark.asyncio
async def test_provider_webhook_config_and_creds_provided_proceeds(tool, session):
    """With required config supplied and an explicit credential, set up the
    preset (config check is satisfied)."""
    graph = _make_graph(
        manual=False,
        regular_credentials={
            "github_credentials": (
                _github_field_info(),
                {("trigger-node", "credentials")},
                True,
            )
        },
        config_schema={
            "properties": {"repo": {"type": "string"}},
            "required": ["repo"],
        },
    )
    preset = _make_preset(
        provider="github",
        url="https://backend.agpt.co/api/integrations/github/webhooks/wh-3/ingress",
    )
    ctxs, setup_mock = _patches(
        graph,
        matched={"github_credentials": _github_meta()},
        available=[_github_credential()],
        preset=preset,
    )
    with ctxs[0], ctxs[1], ctxs[2], ctxs[3], ctxs[4]:
        result = await tool._execute(
            user_id=_USER,
            session=session,
            name="My Trigger",
            graph_id="graph-1",
            trigger_config={"repo": "owner/repo"},
            credentials={"github_credentials": "cred-gh-1"},
        )

    assert isinstance(result, TriggerSetupResponse)
    setup_mock.assert_awaited_once()
    assert setup_mock.await_args.kwargs["trigger_config"] == {"repo": "owner/repo"}


@pytest.mark.asyncio
async def test_setup_failure_surfaces_actionable_message(tool, session):
    """A webhook-setup InvalidInputError (e.g. no enabled events) surfaces its
    specific reason, not a generic error."""
    graph = _make_graph(manual=True)
    ctxs, setup_mock = _patches(graph)
    setup_mock.side_effect = InvalidInputError(
        "Could not set up webhook: Cannot set up github webhook without any "
        "enabled events in event filter input 'events'"
    )
    with ctxs[0], ctxs[1], ctxs[2], ctxs[3], ctxs[4]:
        result = await tool._execute(
            user_id=_USER, session=session, name="My Trigger", graph_id="graph-1"
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "trigger_setup_failed"
    assert "without any enabled events" in result.message
