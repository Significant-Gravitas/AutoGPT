from datetime import datetime
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from backend.copilot.heartbeat import config as hb_config
from backend.copilot.heartbeat.config import (
    HeartbeatConfig,
    checklist_is_empty,
    in_active_hours,
)

_BERLIN = ZoneInfo("Europe/Berlin")


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, 8, hour, minute, tzinfo=_BERLIN)


def test_defaults_match_the_openclaw_heartbeat():
    config = HeartbeatConfig()
    assert config.enabled is False
    assert config.interval_minutes == 30
    assert (config.active_hours_start, config.active_hours_end) == ("08:00", "22:00")
    assert config.model_tier == "standard"
    assert config.delivery.copilot_thread and config.delivery.push
    assert config.delivery.chat_platforms == []


@pytest.mark.parametrize(
    "hour, minute, inside",
    [
        (7, 59, False),
        (8, 0, True),  # the start is inclusive
        (13, 30, True),
        (21, 59, True),
        (22, 0, False),  # the end is exclusive
        (23, 30, False),
    ],
)
def test_a_daytime_window(hour, minute, inside):
    assert in_active_hours(HeartbeatConfig(), _at(hour, minute)) is inside


@pytest.mark.parametrize(
    "hour, inside",
    [(21, False), (22, True), (23, True), (0, True), (5, True), (6, False)],
)
def test_a_window_that_runs_across_midnight(hour, inside):
    night = HeartbeatConfig(active_hours_start="22:00", active_hours_end="06:00")
    assert in_active_hours(night, _at(hour)) is inside


def test_equal_start_and_end_means_all_day():
    always = HeartbeatConfig(active_hours_start="00:00", active_hours_end="00:00")
    assert all(in_active_hours(always, _at(h)) for h in range(24))


@pytest.mark.parametrize(
    "checklist",
    [
        "",
        "   \n\n",
        "# Heartbeat checklist\n\n## Inbox\n",
        "<!-- Add things to check here -->\n# Checks\n- \n- [ ]\n* \n>",
        "<!--\nmulti-line\ncomment\n-->",
    ],
)
def test_a_checklist_with_no_instruction_is_empty(checklist):
    assert checklist_is_empty(checklist)


@pytest.mark.parametrize(
    "checklist",
    [
        "Check whether any agent run failed",
        "# Checks\n- [ ] Did the nightly sync fail?",
        "<!-- note -->\nTell me if a review is waiting",
    ],
)
def test_a_checklist_with_an_instruction_is_not_empty(checklist):
    assert not checklist_is_empty(checklist)


@pytest.mark.parametrize(
    "field, value",
    [
        ("active_hours_start", "8:00"),
        ("active_hours_end", "24:00"),
        ("timezone", "Mars/Olympus"),
        ("interval_minutes", 5),
        ("interval_minutes", 24 * 60 + 1),
        ("model_tier", "turbo"),
    ],
)
def test_invalid_settings_are_refused(field, value):
    with pytest.raises(ValidationError):
        HeartbeatConfig.model_validate({field: value})


async def test_unreadable_stored_settings_read_as_off():
    users = AsyncMock()
    users.get_user_copilot_heartbeat.return_value = {
        "enabled": True,
        "interval_minutes": 1,
    }
    with patch.object(hb_config, "user_db", return_value=users):
        config = await hb_config.load_config("u")
    assert config == HeartbeatConfig()


async def test_stored_settings_round_trip():
    saved = HeartbeatConfig(enabled=True, checklist="Check runs", timezone="UTC")
    users = AsyncMock()
    users.get_user_copilot_heartbeat.return_value = saved.model_dump(mode="json")
    with patch.object(hb_config, "user_db", return_value=users):
        assert await hb_config.load_config("u") == saved


async def test_the_profile_timezone_is_used_when_the_config_has_none():
    users = AsyncMock()
    users.get_user_by_id.return_value = type("U", (), {"timezone": "Asia/Tokyo"})()
    with patch.object(hb_config, "user_db", return_value=users):
        assert await hb_config.resolve_timezone("u", HeartbeatConfig()) == "Asia/Tokyo"
        own = HeartbeatConfig(timezone="Europe/Berlin")
        assert await hb_config.resolve_timezone("u", own) == "Europe/Berlin"
