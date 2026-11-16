from unittest.mock import patch

import pytest
from apscheduler.triggers.cron import CronTrigger

from backend.util.cron import MAX_CRON_LENGTH, validate_cron


def test_validate_cron_refuses_an_overlong_cron_before_parsing_it():
    with (
        patch.object(CronTrigger, "from_crontab") as parse,
        pytest.raises(ValueError, match="too long"),
    ):
        validate_cron("0 9 * * *" + " " * MAX_CRON_LENGTH, "UTC")
    parse.assert_not_called()


def test_validate_cron_returns_an_open_ended_trigger_for_a_cron_that_fires():
    trigger = validate_cron("0 0 29 2 1", "UTC")
    assert trigger.end_date is None
