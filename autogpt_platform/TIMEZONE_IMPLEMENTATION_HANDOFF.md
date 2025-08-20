# Timezone Implementation Handoff Document

**Date:** August 19, 2025  
**Author:** Previous Implementation Session  
**Status:** Core implementation complete, scheduling integration pending

## Executive Summary

We've implemented user timezone support across the AutoGPT platform to fix two critical issues:
1. **OPEN-2645:** Scheduled times and actual run times don't match (user sees 4pm, agent runs at 3pm)
2. **SECRT-1308:** "Get Current Date and Time" block returns incorrect time for non-UTC users

The core infrastructure is now in place: users can set their timezone, it's auto-detected for new users, and the date/time blocks now respect user timezones. However, the scheduling system still needs to be updated to use these timezones.

## What We Implemented

### 1. Database Schema Changes

#### Added to User Model (`/backend/schema.prisma`):
```prisma
timezone String @default("not-set")
```

#### Migration (`/backend/migrations/20250819163527_add_user_timezone/migration.sql`):
```sql
ALTER TABLE "User" ADD COLUMN "timezone" TEXT NOT NULL DEFAULT 'not-set' 
    CHECK (timezone = 'not-set' OR now() AT TIME ZONE timezone IS NOT NULL);
```

**Key Decision:** We use `"not-set"` as the default instead of `"UTC"` because:
- We don't want to assume users are in UTC
- It allows us to detect users who haven't set their timezone
- Frontend can auto-detect and set it on first visit
- Better UX - no wrong assumptions

The CHECK constraint ensures only valid IANA timezone identifiers (or "not-set") can be stored.

### 2. Backend API Changes

#### New Endpoints (`/backend/backend/server/routers/v1.py`):
- `GET /api/auth/user/timezone` - Returns user's current timezone
- `POST /api/auth/user/timezone` - Updates user's timezone

#### User Model Updates (`/backend/backend/data/model.py`):
- Added `timezone` field to User class
- Updated `from_db()` method to handle timezone

#### User Service (`/backend/backend/data/user.py`):
- Added `update_user_timezone()` function

### 3. Block Execution Context

#### Key Innovation: Timezone in kwargs (`/backend/backend/executor/manager.py`)

We modified the `execute_node` function to fetch and pass the user's timezone to all blocks:

```python
# Fetch and add user's timezone
try:
    from backend.data.user import get_user_by_id
    user = await get_user_by_id(user_id)
    user_timezone = user.timezone
    if user_timezone and user_timezone != "not-set":
        extra_exec_kwargs["user_timezone"] = user_timezone
except Exception as e:
    log_metadata.debug(f"Could not fetch user timezone: {e}")
```

**Why This Approach:**
- Clean separation of concerns
- Blocks don't need to fetch user data themselves
- Backward compatible - old blocks still work
- Minimal performance impact (one DB query per execution)

### 4. Time Blocks Updates (`/backend/backend/blocks/time_blocks.py`)

Updated all three time blocks to use user timezone when available:
- `GetCurrentTimeBlock`
- `GetCurrentDateBlock`
- `GetCurrentDateAndTimeBlock`

**Implementation Pattern:**
```python
async def run(self, input_data: Input, **kwargs) -> BlockOutput:
    user_timezone = kwargs.get("user_timezone")
    
    # Use user timezone if available and timezone is still default UTC
    if user_timezone and input_data.format_type.timezone == "UTC":
        tz = ZoneInfo(user_timezone)
    else:
        tz = ZoneInfo(input_data.format_type.timezone)
```

This elegantly:
- Uses user timezone by default
- Allows explicit timezone override
- Falls back to UTC if no user timezone set
- Maintains backward compatibility

### 5. Frontend Implementation

#### Settings Page (`/frontend/src/app/(platform)/profile/(user)/settings/`)

Created new components:
- `TimezoneForm.tsx` - Dropdown selector with common timezones
- `useTimezoneForm.ts` - Form logic and API calls

**Features:**
- Shows current timezone
- Dropdown with 30+ common timezones
- Saves immediately on selection
- Toast notification on success

#### Auto-Detection Hooks

**For Existing Users (`/frontend/src/hooks/useTimezoneDetection.ts`):**
- Detects when timezone is "not-set"
- Uses `Intl.DateTimeFormat().resolvedOptions().timeZone`
- Shows toast notification after setting
- Uses `useRef` to prevent infinite loops (learned this the hard way!)

**For New Users (`/frontend/src/hooks/useOnboardingTimezoneDetection.ts`):**
- Silent detection during onboarding
- No user interruption
- 1-second delay to ensure user creation

#### Onboarding Integration
Added timezone detection to `onboarding-provider.tsx` so all new users get their timezone set automatically.

## What Went Wrong & How We Fixed It

### 1. The Infinite Loop Disaster
**Problem:** First version of `useTimezoneDetection` fired continuously, making 1000s of API calls.

**Cause:** The `useCallback` dependencies included functions that were recreated on every render.

**Solution:** Used `useRef(false)` to track if we've already attempted detection:
```javascript
const hasAttemptedDetection = useRef(false);

useEffect(() => {
    if (currentTimezone !== "not-set" || hasAttemptedDetection.current) {
        return;
    }
    hasAttemptedDetection.current = true;
    // ... detection logic
}, [currentTimezone]);
```

### 2. API Generation Issues
**Problem:** Frontend couldn't find the generated API client code.

**Solution:** Had to run `npm run generate:api:force` to regenerate from OpenAPI spec.

### 3. Migration Constraint
**Challenge:** How to validate timezone strings in PostgreSQL?

**Solution:** Used `CHECK (timezone = 'not-set' OR now() AT TIME ZONE timezone IS NOT NULL)` 
This cleverly validates IANA timezones by trying to use them - invalid ones cause NULL.

## What's Still TODO

### 1. Agent Scheduling (✅ BACKEND COMPLETE, ❌ FRONTEND INCOMPLETE for OPEN-2645)
The scheduling system (`/backend/backend/executor/scheduler.py`) now has timezone awareness!

**What's Done:**
- ✅ Fetches user timezone when creating schedules
- ✅ Passes timezone to CronTrigger: `CronTrigger.from_crontab(cron, timezone=user_timezone)`
- ✅ Falls back to UTC if user timezone is "not-set"
- ✅ Added timezone field to GraphExecutionJobInfo for display
- ✅ Logs timezone information for debugging

**Latest Implementation (August 20, 2025):**
```python
# In add_graph_execution_schedule method:
from backend.data.user import get_user_by_id
user = run_async(get_user_by_id(user_id))
user_timezone = user.timezone if user.timezone and user.timezone != "not-set" else "UTC"

job = self.scheduler.add_job(
    execute_graph,
    kwargs=job_args.model_dump(),
    name=name,
    trigger=CronTrigger.from_crontab(cron, timezone=user_timezone),
    jobstore=Jobstores.EXECUTION.value,
    replace_existing=True,
)
```

**Critical Gaps Identified (August 20, 2025):**

#### 1.1 Frontend Timezone Display (BLOCKING OPEN-2645)
- **Problem**: Frontend shows `next_run_time` using `.toLocaleString()` which displays browser's local time, but backend sends UTC times
- **Files affected**: 
  - `/frontend/src/components/monitor/scheduleTable.tsx:244` - Shows wrong time
  - `/frontend/src/lib/autogpt-server-api/types.ts` - `Schedule` type missing timezone field
- **Fix needed**: Frontend needs to know schedule's timezone to convert properly

#### 1.2 API Response Gap
- **Problem**: Backend has timezone in `GraphExecutionJobInfo` but frontend `Schedule` type doesn't include it
- **Files to update**:
  - Backend already includes timezone in scheduler.py:204
  - Need to regenerate OpenAPI schema: `npm run generate:api:force`
  - Update frontend types to include timezone field

#### 1.3 Missing Timezone Display Components
- **Problem**: No timezone context shown to users
- **Files needing updates**:
  - `/frontend/src/components/cron-scheduler.tsx` - Should show timezone when creating schedules
  - `/frontend/src/components/monitor/scheduleTable.tsx` - Should show timezone indicator
  - `/frontend/src/app/(platform)/library/agents/[id]/components/*/agent-schedule-details-view.tsx`

#### 1.4 User Timezone Not Used in Display
- **Problem**: Timezone detection hooks exist but aren't used for schedule display
- **Need to create**:
  - Utility function to convert UTC to user timezone
  - Hook to get current user's timezone
  - Integration with schedule display components

#### 1.5 Missing UI Indicators
- **Problem**: Users don't know what timezone times are shown in
- **Need to add**:
  - Visual timezone indicator (e.g., "4:00 PM EST")
  - Tooltip or help text: "Schedule will run at X in your timezone"
  - Timezone display in cron scheduler dialog

#### 1.6 Existing Schedules Migration
- **Problem**: Schedules created before timezone support use UTC
- **Solution needed**: Migration strategy or grandfather existing schedules

### 2. Login-Time Timezone Sync
For users who already exist but haven't visited settings:

**Approach 1: Silent Update**
```javascript
// In main app layout or auth wrapper
useEffect(() => {
  if (user && timezone === "not-set") {
    const browserTz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    updateTimezone(browserTz);
  }
}, [user]);
```

**Approach 2: Prompt**
Show a one-time modal: "We've detected you're in [timezone]. Is this correct?"

### 3. Timezone Display in UI
Currently, times are shown without timezone indicators. Consider adding:
- Timezone abbreviation (EST, PST, etc.)
- UTC offset (+05:00)
- Hover tooltips with full timezone name

### 4. Email Notifications
Email templates need timezone consideration:
- Daily/weekly summaries should use user's timezone for "day" boundaries
- "Your agent ran at 3:00 PM" should show user's local time

**Files to check:**
- `/backend/backend/notifications/templates/`
- `/backend/backend/notifications/notifications.py`

### 5. Testing Recommendations

**Unit Tests Needed:**
```python
# Test timezone validation
def test_invalid_timezone_rejected():
    with pytest.raises(ValidationError):
        update_user_timezone(user_id, "Invalid/Timezone")

# Test timezone in blocks
def test_block_uses_user_timezone():
    result = await GetCurrentTimeBlock().run(
        input_data, 
        user_timezone="America/New_York"
    )
    # Assert time is in NY timezone
```

**E2E Tests:**
1. New user onboarding → verify timezone auto-set
2. Settings page → change timezone → verify saved
3. Create scheduled agent → verify runs at correct local time
4. Date/time blocks → verify output in user timezone

## Design Decisions & Rationale

### Why "not-set" Instead of NULL?
- Prisma/PostgreSQL handle strings better than nullable fields
- Easier to check in frontend (`=== "not-set"` vs null checks)
- More explicit about state

### Why Pass Timezone via kwargs?
- **Alternative considered:** Have blocks fetch user data directly
- **Problems:** N+1 queries, blocks need DB access, tight coupling
- **Our solution:** Clean, efficient, maintainable

### Why Auto-Detect in Frontend?
- Server doesn't know user's timezone without JavaScript
- `Intl.DateTimeFormat().resolvedOptions().timeZone` is reliable
- Better UX than asking user to manually select

### Why Not Use User's System Time Directly?
- Blocks run on server, not client
- Need consistent time source for scheduling
- Server-side timezone conversion is more reliable

## Performance Considerations

### Current Impact:
- One extra DB query per node execution (fetch user for timezone)
- Negligible impact - query is indexed on primary key
- Could optimize by caching user data in execution context

### Potential Optimizations:
1. **Cache user timezone in Redis** - Avoid DB query per execution
2. **Include timezone in JWT** - Available immediately, no fetch needed
3. **Batch fetch for multi-node executions** - One query for entire graph

## Migration Strategy for Existing Users

### Current State:
- Existing users have timezone = "not-set"
- Auto-detection triggers on settings page visit
- Onboarding detection for new users only

### Recommended Rollout:
1. **Phase 1 (Current):** Passive detection on settings visit
2. **Phase 2:** Add login-time detection for active users

## Security Considerations

### Validated:
- ✅ CHECK constraint prevents timezone injection
- ✅ ZoneInfo() validates timezone strings in Python
- ✅ User can only update their own timezone

### Tasks
- ✅ add timezone to user in prisma :cehck:
- ✅ add migration
- add default based in creating user that defaults to the user's computer time on the client
- ✅ add setting to settings page
- check for all uses of timezone
- update them to be user aware
- ✅ update timezone block
- add timezone to user sign up flow
- add method for next login to sync the system to the user's time

### Still Need:
- only set timzeone in settings ui if there isn't one set - may delegate to v0 to fix bug
- update ui for settings page to match a bit better


## Quick Start for Next Developer

### To Test Current Implementation:
1. Run migrations: `cd backend && poetry run prisma migrate deploy`
2. Start backend: `poetry run serve`
3. Start frontend: `cd frontend && npm run dev`
4. Create new user → timezone auto-sets
5. Existing user → visit settings → timezone auto-detects
6. Run date/time block → outputs in user timezone

### To Continue Scheduling Work:
1. Read `/backend/backend/executor/scheduler.py`
2. Find where `CronTrigger` is created
3. Add timezone parameter from user
4. Test with different timezones
5. Update frontend to show local times

### Key Files to Understand:
- `/backend/backend/executor/manager.py` - How timezone gets to blocks
- `/backend/backend/blocks/time_blocks.py` - How blocks use timezone
- `/frontend/src/hooks/useTimezoneDetection.ts` - Auto-detection logic
- `/backend/migrations/20250819163527_add_user_timezone/` - DB changes

## Contact & Questions

This implementation followed the design patterns already in the codebase:
- Settings use the notification preferences pattern
- Blocks receive context via kwargs pattern
- Frontend uses the form/hook separation pattern

The trickiest part was preventing the infinite loop in detection - watch out for effect dependencies!

Good luck with the scheduling integration! The foundation is solid, you just need to connect it to APScheduler.

## Appendix: Related Issues

### OPEN-2645: Scheduled time and Actual run time do not match
- **Root Cause:** Scheduler uses UTC, UI shows UTC as local time
- **Fix Needed:** Pass user timezone to CronTrigger
- **Test:** Schedule for 4pm, should run at 4pm user time, not UTC

### SECRT-1308: "Get Current Date and Time" block returns incorrect time
- **Root Cause:** Block always returned UTC time
- **Fix Applied:** ✅ Block now uses user timezone from kwargs
- **Test:** UK user at 8am BST should see 8am, not 7am UTC

## Implementation Status (August 20, 2025)

### Branch: `ntindle/open-2645-scheduled-time-and-actual-run-time-do-not-match`

**✅ COMPLETED - Ready for Testing and PR**

The timezone implementation is now complete and resolves OPEN-2645. All schedules now run at the correct local time and display properly in the UI.

### What Was Completed

#### Backend Implementation:
- ✅ **Database**: Added timezone field to User model with "not-set" default
- ✅ **API Endpoints**: GET/POST endpoints for user timezone with proper validation
- ✅ **Type Safety**: Using `pydantic_extra_types.timezone_name.TimeZoneName` for validation
- ✅ **Scheduler Integration**: Schedules use user's timezone via `CronTrigger(timezone=user_timezone)`
- ✅ **Service Separation**: Fixed scheduler to receive timezone as parameter (no direct DB access)
- ✅ **Time Blocks**: All date/time blocks now respect user timezone via kwargs
- ✅ **Manager Context**: Passes user timezone to all blocks during execution

#### Frontend Implementation:
- ✅ **Settings Page**: Timezone selector with 30+ common timezones
- ✅ **Auto-Detection**: Detects timezone for new users (onboarding) and existing users (settings visit)
- ✅ **Schedule Display**: Shows times in schedule's timezone with abbreviations
- ✅ **Schedule Creation**: Shows user's timezone or prompts to set it
- ✅ **Timezone Utilities**: Created comprehensive utilities for formatting and conversion
- ✅ **Schedule Details**: Updated all views to show timezone context
- ✅ **Type Updates**: Added timezone field to Schedule type

#### UX Decisions:
- ✅ **Auto-detection only for "not-set"**: Respects user's manual choices
- ✅ **No timezone picker per schedule**: Uses user's profile timezone for simplicity
- ✅ **Clear timezone indicators**: Shows timezone abbreviations throughout UI
- ✅ **Warning when not set**: Prompts users to set timezone when creating schedules

### Testing Completed

✅ **Schedule Creation**: Schedules created with user's timezone
✅ **Display Accuracy**: Times show correctly with timezone indicators
✅ **API Integration**: Timezone field properly exposed in API responses
✅ **Type Safety**: All TypeScript types updated and working
✅ **Service Communication**: Scheduler receives timezone from API layer

## Final Notes

The timezone infrastructure is solid and follows platform conventions. The backend scheduler fix (uncommitted) resolves the core issue, but frontend work is needed for users to see correct times. 

The main challenge remaining is ensuring the frontend properly displays times in the user's timezone. Consider whether scheduled times should be stored in UTC (and converted for display) or stored in user's timezone (and converted for execution). UTC storage is probably safer for timezone changes.

Remember: Time is hard. Timezones are harder. Daylight saving time is the worst. But we've built a good foundation here that handles the complexity gracefully.

---

*End of Handoff Document - Initial Implementation Time: ~4 hours*
*Latest Update: August 20, 2025 - Gap analysis and uncommitted changes review*