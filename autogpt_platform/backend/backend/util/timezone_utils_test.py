from unittest.mock import patch

import pydantic
import pytest

from backend.api.external.v1.tools import RunAgentRequest
from backend.api.features.schedules.model import (
    ScheduleCreationRequest,
    ScheduleUpdateRequest,
)
from backend.util.timezone_utils import MAX_TIMEZONE_LENGTH, validate_timezone

LONGEST_ZONE = "America/Argentina/ComodRivadavia"
OVERLONG_ZONE = "Europe/" + "a/" * MAX_TIMEZONE_LENGTH


def test_validate_timezone_refuses_an_overlong_name_without_looking_it_up():
    with patch("backend.util.timezone_utils.ZoneInfo") as lookup:
        assert validate_timezone(OVERLONG_ZONE) is False
    lookup.assert_not_called()


def test_validate_timezone_accepts_the_longest_zone_name():
    assert validate_timezone(LONGEST_ZONE)


@pytest.mark.parametrize(
    "build",
    [
        lambda tz: ScheduleCreationRequest(
            name="s", cron="0 9 * * *", inputs={}, timezone=tz
        ),
        lambda tz: ScheduleUpdateRequest(timezone=tz),
        lambda tz: RunAgentRequest(username_agent_slug="a/b", timezone=tz),
    ],
    ids=["schedule-create", "schedule-update", "v1-run-agent"],
)
def test_request_models_cap_the_timezone_length(build):
    assert build(LONGEST_ZONE).timezone == LONGEST_ZONE
    with pytest.raises(pydantic.ValidationError):
        build(OVERLONG_ZONE)
