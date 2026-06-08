from datetime import datetime, timezone
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest

from backend.api.features.admin import bot_analytics_routes
from backend.data.bot_analytics_reads import (
    BotAnalyticsSummary,
    BotCommandUsage,
    BotGuildInfo,
    BotTimeseriesPoint,
)

app = fastapi.FastAPI()
app.include_router(bot_analytics_routes.router, prefix="/api/admin")
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_admin):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def test_summary_returns_metrics(mocker):
    mock_summary = AsyncMock(
        return_value=BotAnalyticsSummary(
            platform="DISCORD",
            window_days=30,
            live_servers=5,
            messages_received=100,
            replies_sent=90,
            commands_used=10,
            stream_errors=2,
            avg_reply_ms=1200.0,
            error_rate=0.02,
        )
    )
    mocker.patch.object(bot_analytics_routes, "get_bot_analytics_summary", mock_summary)
    resp = client.get("/api/admin/bot-analytics/summary?platform=DISCORD&days=30")
    assert resp.status_code == 200
    data = resp.json()
    assert data["live_servers"] == 5
    assert data["messages_received"] == 100
    assert data["error_rate"] == 0.02
    mock_summary.assert_awaited_once_with("DISCORD", 30)


def test_timeseries_returns_list(mocker):
    mock_timeseries = AsyncMock(
        return_value=[
            BotTimeseriesPoint(
                date=datetime(2026, 6, 1, tzinfo=timezone.utc),
                messages=10,
                replies=9,
                errors=1,
            )
        ]
    )
    mocker.patch.object(
        bot_analytics_routes, "get_bot_message_timeseries", mock_timeseries
    )
    resp = client.get("/api/admin/bot-analytics/timeseries?days=7")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["messages"] == 10
    mock_timeseries.assert_awaited_once_with(None, 7)


def test_command_usage_and_guilds(mocker):
    mock_commands = AsyncMock(return_value=[BotCommandUsage(command="setup", uses=3)])
    mock_guilds = AsyncMock(
        return_value=[
            BotGuildInfo(
                platform="DISCORD",
                server_id="9",
                name="Srv",
                joined_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
                left_at=None,
                active=True,
            )
        ]
    )
    mocker.patch.object(bot_analytics_routes, "get_bot_command_usage", mock_commands)
    mocker.patch.object(bot_analytics_routes, "list_bot_guilds", mock_guilds)

    cmd = client.get("/api/admin/bot-analytics/command-usage")
    assert cmd.status_code == 200
    assert cmd.json()[0]["command"] == "setup"
    mock_commands.assert_awaited_once_with(None, 30)

    guilds = client.get("/api/admin/bot-analytics/guilds")
    assert guilds.status_code == 200
    assert guilds.json()[0]["active"] is True
    mock_guilds.assert_awaited_once_with(None, False, 500)


def test_days_out_of_range_rejected():
    resp = client.get("/api/admin/bot-analytics/summary?days=9999")
    assert resp.status_code == 422


def test_invalid_platform_rejected():
    resp = client.get("/api/admin/bot-analytics/summary?platform=MYSPACE")
    assert resp.status_code == 422
