# Schedule Reset Bug Fix - SECRT-1569

## Problem Summary

A critical bug was identified where schedules set to run "every minute" would reset to run "daily" after the first execution. This affected all minute-based and other timezone-independent recurring schedules.

### Root Cause

The bug was in the `convert_cron_to_utc()` function in `/workspace/autogpt_platform/backend/backend/util/timezone_utils.py`. The function was designed to convert timezone-dependent cron expressions (like "daily at 9 AM") from user timezone to UTC for consistent execution.

However, it was incorrectly converting timezone-independent patterns:

1. **Every minute** (`* * * * *`) would be converted to a specific daily time
2. **Every N minutes** (`*/5 * * * *`) would be converted to specific daily times
3. **Every hour patterns** (`30 * * * *`) would be converted to specific daily times

### The Conversion Process That Caused the Bug

```python
# Original problematic logic:
cron = croniter("* * * * *", now_user)  # Every minute
next_user_time = cron.get_next(datetime)  # Gets next minute, e.g., 14:15
next_utc_time = next_user_time.astimezone(utc_tz)  # Converts to UTC

# Creates new cron with specific time instead of preserving pattern
utc_cron_parts = [
    str(next_utc_time.minute),  # "15" instead of "*"
    str(next_utc_time.hour),    # "14" instead of "*"  
    cron_fields[2],  # "*"
    cron_fields[3],  # "*"
    cron_fields[4],  # "*"
]
# Result: "15 14 * * *" (daily at 14:15) instead of "* * * * *" (every minute)
```

## Solution

Added logic to detect and preserve timezone-independent cron patterns:

### Patterns That Are Now Preserved (No Timezone Conversion)

1. **Every minute**: `* * * * *`
2. **Every N minutes**: `*/5 * * * *`, `*/10 * * * *`, etc.
3. **Every hour at minute M**: `30 * * * *`, `0 * * * *`, etc.
4. **Every N hours**: `0 */2 * * *`, `15 */3 * * *`, etc.

### Patterns That Still Get Timezone Conversion

1. **Daily schedules**: `0 9 * * *` (daily at 9 AM)
2. **Weekly schedules**: `30 14 * * 1` (Mondays at 2:30 PM)
3. **Monthly schedules**: `0 0 1 * *` (1st of month at midnight)
4. **Complex schedules**: `0 12 * * 0,6` (weekends at noon)

## Files Modified

### 1. `/workspace/autogpt_platform/backend/backend/util/timezone_utils.py`

Added timezone-independent pattern detection:

```python
# Every minute: * * * * *
if cron_expr == "* * * * *":
    logger.debug(f"Preserving timezone-independent cron '{cron_expr}' (every minute)")
    return cron_expr
    
# Every N minutes: */N * * * *
if (minute_field.startswith("*/") and 
    hour_field == "*" and 
    cron_fields[2] == "*" and 
    cron_fields[3] == "*" and 
    cron_fields[4] == "*"):
    logger.debug(f"Preserving timezone-independent cron '{cron_expr}' (every N minutes)")
    return cron_expr
    
# ... additional patterns
```

### 2. `/workspace/autogpt_platform/backend/test/test_timezone_utils.py` (New)

Comprehensive test suite covering:
- All timezone-independent patterns are preserved
- Timezone-dependent patterns are still converted
- Specific bug scenario reproduction and verification
- Edge cases and error handling

## Testing

### Bug Scenario Test

```python
def test_minute_schedule_bug_fix(self):
    """Test that the reported bug is fixed."""
    cron_expr = "* * * * *"  # Every minute
    
    for timezone in ["UTC", "America/New_York", "Europe/London", "Asia/Tokyo"]:
        result = convert_cron_to_utc(cron_expr, timezone)
        assert result == cron_expr  # Should be preserved, not converted
```

### Results

✅ Every minute schedules are now preserved across all timezones  
✅ All minute-based intervals (*/N) are preserved  
✅ Hour-based patterns are preserved  
✅ Daily/weekly/monthly patterns still get proper timezone conversion  
✅ No breaking changes to existing functionality  

## Impact

### Fixed Issues
- ✅ Schedules set to "every minute" no longer reset to daily
- ✅ All minute-based recurring tasks now work correctly
- ✅ Hour-based patterns are preserved
- ✅ Users can set reliable high-frequency schedules

### Preserved Functionality
- ✅ Daily schedules still convert properly for user timezones
- ✅ Weekly/monthly schedules work as expected
- ✅ Complex timezone-dependent patterns unchanged

## Deployment Notes

This is a **backward-compatible fix** that:
- Does not require database migrations
- Does not affect existing timezone-dependent schedules
- Only fixes the broken timezone-independent patterns
- Includes comprehensive tests to prevent regression

The fix is ready for immediate deployment to resolve the reported user issue.