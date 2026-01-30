# AutoGPT Test Coverage Analysis

## Executive Summary

The AutoGPT codebase has significant test coverage gaps across all major components.
The overall estimated test coverage is **~15% of backend modules** and **~5% of frontend
modules**. Several security-critical and core infrastructure modules have zero test
coverage. There is no coverage reporting or minimum threshold enforcement in CI.

This document identifies the specific gaps and proposes a prioritized plan for
improvement.

---

## Current State of Testing

### Testing Frameworks in Use

| Component | Framework | Type |
|---|---|---|
| Backend (Python) | pytest, pytest-asyncio, pytest-snapshot | Unit, Integration, Snapshot |
| Frontend (TypeScript) | Playwright | End-to-End only |
| Frontend (TypeScript) | Storybook + Jest | Visual/Component (minimal) |
| Shared Libs (Python) | pytest, pytest-mock | Unit only |
| Classic (Python) | pytest + VCR | Unit, Integration |

### Test File Counts

| Area | Source Modules | Modules with Tests | Coverage |
|---|---|---|---|
| Backend - Blocks | 162 | 1 (+ centralized parametrized test) | ~1% unit, ~25% via parametrized |
| Backend - Server | 36 | 15 | 42% |
| Backend - Server V2 | 20 | 10 | 50% |
| Backend - Integrations | 17 | 0 | **0%** |
| Backend - Data Layer | 19 | 3 | 16% |
| Backend - Executor | 5 | 3 | 60% |
| Backend - Utilities | 24 | 10 | 42% |
| Backend - Notifications | 2 | 0 | **0%** |
| Backend - Monitoring | 3 | 0 | **0%** |
| Shared Libs (autogpt_libs) | 25 | 3 | 12% |
| Frontend - Components | 215 | 1 | **0.5%** |
| Frontend - Libraries | 23 | 0 | **0%** |
| Frontend - Hooks | 6 | 0 | **0%** |
| Frontend - Pages/Routes | 45 | ~14 (E2E) | ~31% |

### CI/CD Testing Infrastructure

- **Backend CI**: `platform-backend-ci.yml` runs `poetry run pytest -s -vv`
- **Frontend CI**: Playwright E2E tests run against Chromium (headless)
- **Coverage reporting**: Commented out (Codecov action disabled)
- **Coverage thresholds**: None configured
- **No `.coveragerc`** or `[tool.coverage]` section in `pyproject.toml`

---

## Critical Coverage Gaps

### 1. Integrations Module (0% coverage) - CRITICAL

**Path**: `backend/backend/integrations/`

The entire integrations directory has zero test coverage. This includes:

**OAuth handlers** (`integrations/oauth/`):
- `base.py` - Base OAuth handler with token refresh logic, expiry checking
- `github.py` - GitHub OAuth
- `google.py` - Google OAuth
- `notion.py` - Notion OAuth
- `todoist.py` - Todoist OAuth
- `twitter.py` - Twitter OAuth

**Webhook handlers** (`integrations/webhooks/`):
- `_base.py` - Base webhook handler
- `_manual_base.py` - Manual webhook base
- `github.py` - GitHub webhook processing
- `compass.py`, `slant3d.py` - Third-party webhooks
- `graph_lifecycle_hooks.py` - Graph lifecycle event hooks
- `utils.py` - Webhook utilities

**Credential management**:
- `credentials_store.py` - Stores and retrieves user credentials
- `creds_manager.py` - Manages credential lifecycle
- `providers.py` - Provider definitions

**Why this matters**: OAuth and credential handling are security-critical paths. Bugs
here could lead to token leaks, failed authentication flows, or credential storage
issues. The `BaseOAuthHandler.needs_refresh()` method at `integrations/oauth/base.py:73`
contains time-based logic that is easy to get wrong and should have explicit test cases.

**Recommended tests**:
- Unit tests for `BaseOAuthHandler.needs_refresh()` with various time scenarios
- Unit tests for `BaseOAuthHandler.refresh_tokens()` provider validation
- Tests for each OAuth provider's login URL generation and token exchange
- Tests for webhook signature verification
- Tests for credential store read/write operations

---

### 2. Notification System (0% coverage) - CRITICAL

**Path**: `backend/backend/notifications/`

Both files in the notifications module are untested:

- `notifications.py` (920+ lines) - `NotificationManager` service with RabbitMQ-based
  queue processing, batch notification logic, email dispatch, and summary generation
- `email.py` - Email sending via Postmark

**Why this matters**: The notification manager contains complex business logic including:
- Batch processing with size-aware chunking (`_process_batch`, line 612-759)
- User preference checking before sending
- Dead letter queue handling
- Weekly/daily summary data gathering and formatting
- Dynamic email chunk sizing to stay under 4.5MB limits

These are exactly the kind of functions where subtle bugs hide and are hard to catch
without automated tests.

**Recommended tests**:
- Unit tests for `get_routing_key()` with each `NotificationType`
- Unit tests for `_parse_message()` with valid and malformed JSON
- Tests for `_should_email_user_based_on_preference()` with various preference states
- Tests for batch chunking logic in `_process_batch()`
- Tests for `_gather_summary_data()` with mock database responses
- Tests for `create_notification_config()` queue/exchange setup

---

### 3. Monitoring Module (0% coverage) - HIGH

**Path**: `backend/backend/monitoring/`

All three monitoring modules are untested:
- `block_error_monitor.py` - Monitors block execution errors
- `late_execution_monitor.py` - Detects executions that are running late
- `notification_monitor.py` - Monitors notification delivery

**Why this matters**: Monitoring systems are the safety net for production issues. If
monitoring code itself has bugs, real problems can go undetected.

**Recommended tests**:
- Tests for error detection thresholds and alerting logic
- Tests for late execution detection with various timing scenarios
- Tests for notification delivery tracking

---

### 4. Data Layer (84% uncovered) - HIGH

**Path**: `backend/backend/data/`

Only 3 of 19 modules have tests (`credit_test.py`, `graph_test.py`, `model_test.py`).

**Untested modules**:
- `db.py` - Core database connection management, transaction handling, advisory locks
- `api_key.py` - API key creation, validation, and management
- `execution.py` - Execution record management
- `user.py` - User data operations
- `analytics.py` - Analytics data aggregation
- `event_bus.py` - Event publication/subscription
- `rabbitmq.py` - RabbitMQ client wrapper
- `redis_client.py` - Redis client wrapper
- `notifications.py` - Notification data models and queries
- `block.py`, `block_cost_config.py`, `cost.py` - Block and cost data
- `integrations.py` - Integration data layer
- `onboarding.py` - Onboarding flow data

**Why this matters**: The data layer is the foundation of the entire application. The
`db.py` module at `backend/data/db.py:90-134` contains transaction and advisory lock
logic (`locked_transaction`) that is subtle and easy to break. The `api_key.py` module
handles security-sensitive operations.

**Recommended tests**:
- Tests for `db.py`: `add_param()` URL manipulation, `BaseDbModel` ID generation,
  `get_database_schema()` parsing
- Tests for `api_key.py`: Key creation, hashing, and validation
- Tests for `execution.py`: CRUD operations with mock Prisma client
- Tests for `user.py`: User lookup, creation, preference management
- Tests for `event_bus.py`: Event publish/subscribe mechanics

---

### 5. Auth Middleware in Shared Libs (0% coverage) - HIGH

**Path**: `autogpt_libs/autogpt_libs/auth/middleware.py`

The authentication middleware (`auth_middleware` function and `APIKeyValidator` class)
has no test coverage despite being a security-critical component.

**What's untested**:
- `auth_middleware()` - JWT token validation for every API request
- `APIKeyValidator` - API key validation with custom validation function support
- `APIKeyValidator.__call__()` - The actual validation logic including async support
- `APIKeyValidator.default_validator()` - Timing-safe token comparison

**Also untested in the same package**:
- `jwt_utils.py` - JWT token parsing
- `rate_limit/limiter.py` - Redis-based rate limiting
- `rate_limit/middleware.py` - Rate limit middleware
- `api_key/key_manager.py` - API key generation and hashing

**Recommended tests**:
- Test `auth_middleware` with valid JWT, expired JWT, missing header, auth disabled
- Test `APIKeyValidator` with correct key, wrong key, missing key, custom validator
- Test `APIKeyValidator` with both sync and async custom validators
- Test rate limiter with Redis mock: under limit, at limit, over limit
- Test API key manager: generation, hashing, verification

---

### 6. Block Integration Modules (99% uncovered) - MEDIUM-HIGH

**Path**: `backend/backend/blocks/` (subdirectories)

There are 122+ integration block modules across 25+ third-party service directories
(airtable, github, google, twitter, discord, etc.) with only 1 test file
(`airtable/_api_test.py`).

While the root-level blocks get some coverage from the centralized parametrized test in
`blocks/test/test_block.py`, the integration-specific modules (API clients, OAuth
configs, webhook handlers, data mappers) are completely untested.

**Key untested areas**:
- GitHub: Issues, PRs, checks, CI, reviews, triggers (10 modules)
- Google: Calendar, Gmail, Sheets, Auth (4 modules)
- Twitter/X: DMs, lists, spaces, tweets, users (22+ modules)
- Airtable: Records, schema, triggers, webhooks (7 modules)
- Firecrawl: Crawl, extract, map, scrape, search (7 modules)
- All social media posting via Ayrshare (14 modules)

**Recommended approach**:
- Start with the most-used integrations (GitHub, Google, Twitter)
- Focus on data transformation and API response parsing logic
- Mock external API calls; test request construction and response handling
- Test error handling paths for API failures and rate limits

---

### 7. Frontend Unit Testing (0% coverage) - MEDIUM-HIGH

**Path**: `autogpt_platform/frontend/src/`

The frontend has **no unit testing framework configured**. There is one commented-out
test file (`Badge.test.tsx`). The only testing is through Playwright E2E tests, which
cover approximately 31% of pages/routes.

**Critical untested frontend areas**:

**Core workflow editor** (0 unit tests):
- `Flow.tsx` - Main graph editor component
- `CustomNode.tsx` - Custom workflow node rendering
- `CustomEdge.tsx` - Custom edge rendering
- `ConnectionLine.tsx` - Connection line rendering

**API client layer** (0 tests):
- `lib/autogpt-server-api/client.ts` - The primary API client
- `lib/autogpt-server-api/helpers.ts` - API helper functions
- `lib/autogpt-server-api/utils.ts` - API utilities

**Authentication layer** (0 unit tests):
- `lib/supabase/actions.ts` - Server actions for auth
- `lib/supabase/helpers.ts` - Supabase helper functions
- `lib/supabase/middleware.ts` - Auth middleware
- `lib/withRoleAccess.ts` - Role-based access control

**Custom hooks** (0 tests):
- `hooks/useAgentGraph.tsx` - Core workflow state management
- `hooks/useCredentials.ts` - Credential management
- `hooks/useCredits.ts` - Credit/billing management
- `hooks/useCopyPaste.ts` - Clipboard operations

**E2E gaps** (pages without Playwright coverage):
- All admin pages (`/admin/dashboard`, `/admin/marketplace`, `/admin/settings`,
  `/admin/spending`, `/admin/users`)
- Reset password page
- Health check endpoint
- API proxy routes

**Recommended approach**:
1. Set up Vitest (already partially configured given the Badge test imports)
2. Start with utility functions in `lib/` - pure functions are easiest to test
3. Add hook tests using `@testing-library/react-hooks`
4. Add component tests for core workflow components using React Testing Library
5. Expand E2E coverage to admin pages

---

### 8. Server V2 API Gaps - MEDIUM

**Path**: `backend/backend/server/v2/`

10 of 20 modules in the V2 API are untested:

- `AutoMod/manager.py` - Content moderation management
- `AutoMod/models.py` - Moderation data models
- `admin/store_admin_routes.py` - Admin store management endpoints
- `library/routes/agents.py` - Agent management routes
- `library/routes/presets.py` - Preset management routes
- `otto/models.py` - Otto AI assistant models
- `otto/service.py` - Otto AI service logic
- `store/exceptions.py` - Store exception definitions
- `store/image_gen.py` - Image generation service
- `turnstile/models.py` - CAPTCHA models

**Recommended tests**:
- Route handler tests for `admin/store_admin_routes.py` and `library/routes/`
- Service logic tests for `otto/service.py`
- Model validation tests for all model files

---

## Infrastructure Gaps

### No Coverage Reporting

The Codecov integration is commented out in CI:
```yaml
# - name: Upload coverage reports to Codecov
#   uses: codecov/codecov-action@v4
```

**Recommendation**: Re-enable coverage reporting and set a baseline threshold. Even a
low initial threshold (e.g., 20%) prevents further regression while the team works
toward higher coverage.

### No Coverage Configuration

There is no `[tool.coverage]` section in `pyproject.toml` and no `.coveragerc` file.

**Recommendation**: Add coverage configuration:
```toml
[tool.coverage.run]
source = ["backend"]
omit = ["*/test/*", "*_test.py", "*/conftest.py"]

[tool.coverage.report]
fail_under = 20
show_missing = true
```

### Inconsistent Test File Naming

The codebase uses three different test file naming conventions:
- `test_*.py` (prefix style)
- `*_test.py` (suffix style)
- `*_tests.py` (plural suffix style)

**Recommendation**: Standardize on one convention. The suffix style (`*_test.py`) is
already the most common in this codebase and works well with collocated tests.

### No Frontend Unit Test Framework

Despite having a Storybook setup with Jest test runner configuration, there is no
functional unit testing framework for React components, hooks, or utility functions.

**Recommendation**: Configure Vitest with React Testing Library. The existing
`Badge.test.tsx` suggests this was planned but never completed.

---

## Prioritized Improvement Plan

### Phase 1: Security-Critical (Immediate)

1. **Auth middleware tests** (`autogpt_libs/auth/middleware.py`, `jwt_utils.py`)
   - JWT validation, token expiry, missing credentials, auth bypass when disabled
   - API key validation with timing-safe comparison

2. **OAuth handler tests** (`backend/integrations/oauth/`)
   - Token refresh logic, expiry detection, provider validation
   - Each provider's login URL and token exchange

3. **API key management tests** (`autogpt_libs/api_key/key_manager.py`)
   - Key generation, hashing, verification

4. **Credential store tests** (`backend/integrations/credentials_store.py`)
   - Secure credential storage and retrieval

5. **Rate limiter tests** (`autogpt_libs/rate_limit/`)
   - Under/at/over limit scenarios, Redis failure handling

### Phase 2: Core Infrastructure (Short-term)

6. **Data layer tests** (`backend/data/db.py`, `api_key.py`, `execution.py`, `user.py`)
   - Database connection management, transaction logic, advisory locks
   - CRUD operations for key entities

7. **Notification system tests** (`backend/notifications/`)
   - Routing key generation, message parsing, batch processing
   - User preference checking, email dispatch

8. **Monitoring tests** (`backend/monitoring/`)
   - Error detection, late execution alerts, notification tracking

9. **Enable CI coverage reporting**
   - Re-enable Codecov, set baseline threshold at current coverage level

### Phase 3: Business Logic (Medium-term)

10. **Server V2 route tests** for untested endpoints
11. **Block integration tests** for top 5 most-used integrations
12. **Frontend utility tests** - Set up Vitest, test `lib/` functions
13. **Frontend hook tests** - `useAgentGraph`, `useCredentials`

### Phase 4: Comprehensive (Long-term)

14. **Frontend component tests** - Core workflow editor components
15. **Remaining block integration tests**
16. **Frontend E2E expansion** - Admin pages, error states
17. **Performance/load tests** for notification batching and execution engine

---

## Metrics to Track

| Metric | Current (Est.) | Phase 1 Target | Phase 2 Target | Phase 4 Target |
|---|---|---|---|---|
| Backend line coverage | ~15% | 25% | 40% | 60% |
| Frontend unit coverage | 0% | 5% | 15% | 30% |
| Frontend E2E page coverage | 31% | 35% | 50% | 70% |
| Security-critical module coverage | 0% | 80% | 90% | 95% |
| CI coverage gate | None | Enabled | 30% min | 50% min |
