"""Tests for ListAgentTriggersTool — focuses on webhook URL exposure."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.list_agent_triggers import (
    AgentTriggerListResponse,
    ListAgentTriggersTool,
)
from backend.copilot.tools.models import ErrorResponse

from ._test_data import make_session

_USER = "test-user-triggers-list"
_PATH = "backend.copilot.tools.list_agent_triggers"


@pytest.fixture
def tool():
    return ListAgentTriggersTool()


@pytest.fixture
def session():
    return make_session(_USER)


def _make_webhook_preset():
    preset = MagicMock()
    preset.id = "preset-1"
    preset.name = "Incoming webhook"
    preset.description = "fires on HTTP event"
    preset.is_active = True
    preset.webhook_id = "wh-1"
    preset.webhook.url = (
        "https://backend.agpt.co/api/integrations/generic_webhook"
        "/webhooks/wh-1/ingress"
    )
    preset.webhook.provider = "generic_webhook"
    return preset


@pytest.mark.asyncio
async def test_list_triggers_no_auth(tool, session):
    result = await tool._execute(user_id=None, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error == "auth_required"


@pytest.mark.asyncio
async def test_list_triggers_exposes_webhook_url(tool, session):
    preset_response = MagicMock()
    preset_response.presets = [_make_webhook_preset()]

    mock_ldb = MagicMock()
    mock_ldb.get_library_agent = AsyncMock(return_value=MagicMock(graph_id="graph-1"))
    mock_ldb.list_trigger_agents = AsyncMock(return_value=[])
    mock_ldb.list_presets = AsyncMock(return_value=preset_response)

    with patch(f"{_PATH}.library_db", return_value=mock_ldb):
        result = await tool._execute(
            user_id=_USER, session=session, library_agent_id="lib-1"
        )

    assert isinstance(result, AgentTriggerListResponse)
    assert len(result.triggers) == 1
    trigger = result.triggers[0]
    assert trigger.kind == "webhook"
    assert trigger.webhook_id == "wh-1"
    assert trigger.webhook_url and trigger.webhook_url.endswith("/ingress")
    assert "backend.agpt.co" in trigger.webhook_url
    assert trigger.provider == "generic_webhook"


@pytest.mark.asyncio
async def test_list_triggers_includes_trigger_agents(tool, session):
    """The 'agent' kind (hidden trigger agents) is surfaced alongside webhooks."""
    trigger_agent = MagicMock()
    trigger_agent.id = "lib-ta"
    trigger_agent.name = "Daily Fetcher"
    trigger_agent.description = "fetches daily"
    trigger_agent.graph_id = "graph-ta"
    trigger_agent.is_scheduled = True
    trigger_agent.next_scheduled_run = "2026-06-09T00:00:00Z"

    preset_response = MagicMock()
    preset_response.presets = []

    mock_ldb = MagicMock()
    mock_ldb.get_library_agent = AsyncMock(return_value=MagicMock(graph_id="graph-1"))
    mock_ldb.list_trigger_agents = AsyncMock(return_value=[trigger_agent])
    mock_ldb.list_presets = AsyncMock(return_value=preset_response)

    with patch(f"{_PATH}.library_db", return_value=mock_ldb):
        result = await tool._execute(
            user_id=_USER, session=session, library_agent_id="lib-1"
        )

    assert isinstance(result, AgentTriggerListResponse)
    assert len(result.triggers) == 1
    trigger = result.triggers[0]
    assert trigger.kind == "agent"
    assert trigger.id == "lib-ta"
    assert trigger.is_scheduled is True
    assert trigger.graph_id == "graph-ta"
