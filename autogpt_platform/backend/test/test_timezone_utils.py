"""
Tests for timezone utility functions, specifically the schedule reset bug fix.
"""

import pytest
from unittest.mock import patch
from datetime import datetime
from zoneinfo import ZoneInfo

from backend.util.timezone_utils import convert_cron_to_utc, convert_utc_time_to_user_timezone


class TestConvertCronToUtc:
    """Test the convert_cron_to_utc function."""
    
    def test_every_minute_preserved(self):
        """Test that 'every minute' pattern is preserved across all timezones."""
        cron_expr = "* * * * *"
        timezones = ["UTC", "America/New_York", "Europe/London", "Asia/Tokyo"]
        
        for timezone in timezones:
            result = convert_cron_to_utc(cron_expr, timezone)
            assert result == cron_expr, f"Every minute should be preserved in {timezone}"
    
    def test_every_n_minutes_preserved(self):
        """Test that 'every N minutes' patterns are preserved."""
        test_cases = ["*/5 * * * *", "*/10 * * * *", "*/15 * * * *", "*/30 * * * *"]
        timezones = ["UTC", "America/New_York", "Europe/London", "Asia/Tokyo"]
        
        for cron_expr in test_cases:
            for timezone in timezones:
                result = convert_cron_to_utc(cron_expr, timezone)
                assert result == cron_expr, f"{cron_expr} should be preserved in {timezone}"
    
    def test_every_hour_at_minute_preserved(self):
        """Test that 'every hour at minute M' patterns are preserved."""
        test_cases = ["0 * * * *", "15 * * * *", "30 * * * *", "45 * * * *"]
        timezones = ["UTC", "America/New_York", "Europe/London", "Asia/Tokyo"]
        
        for cron_expr in test_cases:
            for timezone in timezones:
                result = convert_cron_to_utc(cron_expr, timezone)
                assert result == cron_expr, f"{cron_expr} should be preserved in {timezone}"
    
    def test_every_n_hours_preserved(self):
        """Test that 'every N hours' patterns are preserved."""
        test_cases = ["0 */2 * * *", "30 */3 * * *", "15 */6 * * *", "0 */12 * * *"]
        timezones = ["UTC", "America/New_York", "Europe/London", "Asia/Tokyo"]
        
        for cron_expr in test_cases:
            for timezone in timezones:
                result = convert_cron_to_utc(cron_expr, timezone)
                assert result == cron_expr, f"{cron_expr} should be preserved in {timezone}"
    
    @patch('backend.util.timezone_utils.datetime')
    def test_daily_patterns_converted(self, mock_datetime):
        """Test that daily patterns are still timezone-converted."""
        # Mock datetime.now() to return a consistent time for testing
        mock_now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        mock_datetime.now.return_value = mock_now
        
        cron_expr = "0 9 * * *"  # Daily at 9 AM
        
        # Should be different when converted from non-UTC timezone
        result = convert_cron_to_utc(cron_expr, "America/New_York")
        assert result != cron_expr, "Daily pattern should be converted from non-UTC timezone"
        
        # Should be same when already in UTC
        result_utc = convert_cron_to_utc(cron_expr, "UTC")
        assert result_utc == cron_expr, "Daily pattern should remain same in UTC"
    
    def test_invalid_cron_expression(self):
        """Test that invalid cron expressions raise ValueError."""
        invalid_expressions = [
            "* * *",  # Too few fields
            "* * * * * *",  # Too many fields
            "",  # Empty string
        ]
        
        for invalid_expr in invalid_expressions:
            with pytest.raises(ValueError):
                convert_cron_to_utc(invalid_expr, "UTC")
    
    def test_invalid_timezone(self):
        """Test that invalid timezones raise ValueError."""
        with pytest.raises(ValueError):
            convert_cron_to_utc("* * * * *", "Invalid/Timezone")


class TestBugScenario:
    """Test the specific bug scenario reported in SECRT-1569."""
    
    def test_minute_schedule_bug_fix(self):
        """
        Test that the reported bug is fixed:
        - Schedule set to run every minute should not reset to daily after first run
        """
        # This is the exact scenario from the bug report
        cron_expr = "* * * * *"  # Every minute
        
        # Test with various timezones that users might have
        timezones = [
            "UTC",
            "America/New_York", 
            "America/Los_Angeles",
            "Europe/London",
            "Europe/Paris", 
            "Asia/Tokyo",
            "Australia/Sydney"
        ]
        
        for timezone in timezones:
            # The conversion should preserve the original expression
            result = convert_cron_to_utc(cron_expr, timezone)
            
            assert result == cron_expr, (
                f"BUG REPRODUCED: Every minute schedule in {timezone} "
                f"was converted to '{result}' instead of being preserved as '{cron_expr}'"
            )
    
    def test_other_minute_intervals_bug_fix(self):
        """Test that other minute-based intervals are also preserved."""
        minute_intervals = [
            "*/2 * * * *",   # Every 2 minutes
            "*/3 * * * *",   # Every 3 minutes  
            "*/5 * * * *",   # Every 5 minutes
            "*/10 * * * *",  # Every 10 minutes
            "*/15 * * * *",  # Every 15 minutes
        ]
        
        for cron_expr in minute_intervals:
            for timezone in ["America/New_York", "Europe/London", "Asia/Tokyo"]:
                result = convert_cron_to_utc(cron_expr, timezone)
                assert result == cron_expr, (
                    f"Minute interval {cron_expr} in {timezone} should be preserved, "
                    f"but was converted to '{result}'"
                )


if __name__ == "__main__":
    # Run the tests directly if this file is executed
    pytest.main([__file__, "-v"])