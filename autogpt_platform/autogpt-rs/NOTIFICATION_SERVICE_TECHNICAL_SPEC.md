# Notification Service Technical Specification

## Overview

The AutoGPT Platform Notification Service is a RabbitMQ-based asynchronous notification system that handles various types of user notifications including real-time alerts, batched notifications, and scheduled summaries. The service supports email delivery via Postmark and system alerts via Discord.

## Architecture Overview

### Core Components

1. **NotificationManager Service** (`notifications.py`)
   - AppService implementation with RabbitMQ integration
   - Processes notification queues asynchronously
   - Manages batching strategies and delivery timing
   - Handles email templating and sending

2. **RabbitMQ Message Broker**
   - Multiple queues for different notification strategies
   - Dead letter exchange for failed messages
   - Topic-based routing for message distribution

3. **Email Sender** (`email.py`)
   - Postmark integration for email delivery
   - Jinja2 template rendering
   - HTML email composition with unsubscribe headers

4. **Database Storage**
   - Notification batching tables
   - User preference storage
   - Email verification tracking

## Service Exposure Mechanism

### AppService Framework

The NotificationManager extends `AppService` which automatically exposes methods decorated with `@expose` as HTTP endpoints:

```python
class NotificationManager(AppService):
    @expose
    def queue_weekly_summary(self):
        # Implementation
    
    @expose
    def process_existing_batches(self, notification_types: list[NotificationType]):
        # Implementation
    
    @expose
    async def discord_system_alert(self, content: str):
        # Implementation
```

### Automatic HTTP Endpoint Creation

When the service starts, the AppService base class:
1. Scans for methods with `@expose` decorator
2. Creates FastAPI routes for each exposed method:
   - Route path: `/{method_name}`
   - HTTP method: POST
   - Endpoint handler: Generated via `_create_fastapi_endpoint()`

### Service Client Access

#### NotificationManagerClient
```python
class NotificationManagerClient(AppServiceClient):
    @classmethod
    def get_service_type(cls):
        return NotificationManager
    
    # Direct method references (sync)
    process_existing_batches = NotificationManager.process_existing_batches
    queue_weekly_summary = NotificationManager.queue_weekly_summary
    
    # Async-to-sync conversion
    discord_system_alert = endpoint_to_sync(NotificationManager.discord_system_alert)
```

#### Client Usage Pattern
```python
# Get client instance
client = get_service_client(NotificationManagerClient)

# Call exposed methods via HTTP
client.process_existing_batches([NotificationType.AGENT_RUN])
client.queue_weekly_summary()
client.discord_system_alert("System alert message")
```

### HTTP Communication Details

1. **Service URL**: `http://{host}:{notification_service_port}`
   - Default port: 8007
   - Host: Configurable via settings

2. **Request Format**:
   - Method: POST
   - Path: `/{method_name}`
   - Body: JSON with method parameters

3. **Client Implementation**:
   - Uses `httpx` for HTTP requests
   - Automatic retry on connection failures
   - Configurable timeout (default from api_call_timeout)

### Direct Function Calls

The service also exposes two functions that can be called directly without going through the service client:

```python
# Sync version - used by ExecutionManager
def queue_notification(event: NotificationEventModel) -> NotificationResult

# Async version - used by credit system
async def queue_notification_async(event: NotificationEventModel) -> NotificationResult
```

These functions:
- Connect directly to RabbitMQ
- Publish messages to appropriate queues
- Return success/failure status
- Are NOT exposed via HTTP

## Message Queuing Architecture

### RabbitMQ Configuration

#### Exchanges
```python
NOTIFICATION_EXCHANGE = Exchange(name="notifications", type=ExchangeType.TOPIC)
DEAD_LETTER_EXCHANGE = Exchange(name="dead_letter", type=ExchangeType.TOPIC)
```

#### Queues
1. **immediate_notifications**
   - Routing Key: `notification.immediate.#`
   - Dead Letter: `failed.immediate`
   - For: Critical alerts, errors

2. **admin_notifications**
   - Routing Key: `notification.admin.#`
   - Dead Letter: `failed.admin`
   - For: Refund requests, system alerts

3. **summary_notifications**
   - Routing Key: `notification.summary.#`
   - Dead Letter: `failed.summary`
   - For: Daily/weekly summaries

4. **batch_notifications**
   - Routing Key: `notification.batch.#`
   - Dead Letter: `failed.batch`
   - For: Agent runs, batched events

5. **failed_notifications**
   - Routing Key: `failed.#`
   - For: All failed messages

### Queue Strategies (QueueType enum)

1. **IMMEDIATE**: Send right away (errors, critical notifications)
2. **BATCH**: Batch for configured delay (agent runs)
3. **SUMMARY**: Scheduled digest (daily/weekly summaries)
4. **BACKOFF**: Exponential backoff strategy (defined but not fully implemented)
5. **ADMIN**: Admin-only notifications

## Notification Types

### Enum Values (NotificationType)
```python
AGENT_RUN                  # Batch strategy, 1 day delay
ZERO_BALANCE              # Backoff strategy, 60 min delay
LOW_BALANCE               # Immediate strategy
BLOCK_EXECUTION_FAILED    # Backoff strategy, 60 min delay
CONTINUOUS_AGENT_ERROR    # Backoff strategy, 60 min delay
DAILY_SUMMARY             # Summary strategy
WEEKLY_SUMMARY            # Summary strategy
MONTHLY_SUMMARY           # Summary strategy
REFUND_REQUEST            # Admin strategy
REFUND_PROCESSED          # Admin strategy
```

## Integration Points

### 1. Scheduler Integration
The scheduler service (`backend.executor.scheduler`) imports monitoring functions that call the NotificationManagerClient:

```python
from backend.monitoring import (
    process_existing_batches,
    process_weekly_summary,
)

# These are scheduled as cron jobs
```

### 2. Execution Manager Integration
The ExecutionManager directly calls `queue_notification()` for:
- Agent run completions
- Low balance alerts

```python
from backend.notifications.notifications import queue_notification

# Called after graph execution completes
queue_notification(NotificationEventModel(
    user_id=graph_exec.user_id,
    type=NotificationType.AGENT_RUN,
    data=AgentRunData(...)
))
```

### 3. Credit System Integration
The credit system uses `queue_notification_async()` for:
- Refund requests
- Refund processed notifications

```python
from backend.notifications.notifications import queue_notification_async

await queue_notification_async(NotificationEventModel(
    user_id=user_id,
    type=NotificationType.REFUND_REQUEST,
    data=RefundRequestData(...)
))
```

### 4. Monitoring Module Wrappers
The monitoring module provides wrapper functions that are used by the scheduler:

```python
# backend/monitoring/notification_monitor.py
def process_existing_batches(**kwargs):
    args = NotificationJobArgs(**kwargs)
    get_notification_manager_client().process_existing_batches(
        args.notification_types
    )

def process_weekly_summary(**kwargs):
    get_notification_manager_client().queue_weekly_summary()
```

## Data Models

### Base Event Model
```typescript
interface BaseEventModel {
  type: NotificationType;
  user_id: string;
  created_at: string; // ISO datetime with timezone
}
```

### Notification Event Model
```typescript
interface NotificationEventModel<T> extends BaseEventModel {
  data: T;
}
```

### Notification Data Types

#### AgentRunData
```typescript
interface AgentRunData {
  agent_name: string;
  credits_used: number;
  execution_time: number;
  node_count: number;
  graph_id: string;
  outputs: Array<Record<string, any>>;
}
```

#### ZeroBalanceData
```typescript
interface ZeroBalanceData {
  last_transaction: number;
  last_transaction_time: string; // ISO datetime with timezone
  top_up_link: string;
}
```

#### LowBalanceData
```typescript
interface LowBalanceData {
  agent_name: string;
  current_balance: number; // credits (100 = $1)
  billing_page_link: string;
  shortfall: number;
}
```

#### BlockExecutionFailedData
```typescript
interface BlockExecutionFailedData {
  block_name: string;
  block_id: string;
  error_message: string;
  graph_id: string;
  node_id: string;
  execution_id: string;
}
```

#### ContinuousAgentErrorData
```typescript
interface ContinuousAgentErrorData {
  agent_name: string;
  error_message: string;
  graph_id: string;
  execution_id: string;
  start_time: string; // ISO datetime with timezone
  error_time: string; // ISO datetime with timezone
  attempts: number;
}
```

#### Summary Data Types
```typescript
interface BaseSummaryData {
  total_credits_used: number;
  total_executions: number;
  most_used_agent: string;
  total_execution_time: number;
  successful_runs: number;
  failed_runs: number;
  average_execution_time: number;
  cost_breakdown: Record<string, number>;
}

interface DailySummaryData extends BaseSummaryData {
  date: string; // ISO datetime with timezone
}

interface WeeklySummaryData extends BaseSummaryData {
  start_date: string; // ISO datetime with timezone
  end_date: string; // ISO datetime with timezone
}
```

#### RefundRequestData
```typescript
interface RefundRequestData {
  user_id: string;
  user_name: string;
  user_email: string;
  transaction_id: string;
  refund_request_id: string;
  reason: string;
  amount: number;
  balance: number;
}
```

### Summary Parameters
```typescript
interface BaseSummaryParams {
  start_date: string; // ISO datetime with timezone
  end_date: string; // ISO datetime with timezone
}

interface DailySummaryParams extends BaseSummaryParams {
  date: string; // ISO datetime with timezone
}

interface WeeklySummaryParams extends BaseSummaryParams {
  start_date: string; // ISO datetime with timezone
  end_date: string; // ISO datetime with timezone
}
```

## Database Schema

### NotificationEvent Table
```sql
model NotificationEvent {
  id        String   @id @default(uuid())
  createdAt DateTime @default(now())
  updatedAt DateTime @default(now()) @updatedAt
  UserNotificationBatch   UserNotificationBatch? @relation
  userNotificationBatchId String?
  type NotificationType
  data Json
  @@index([userNotificationBatchId])
}
```

### UserNotificationBatch Table
```sql
model UserNotificationBatch {
  id        String   @id @default(uuid())
  createdAt DateTime @default(now())
  updatedAt DateTime @default(now()) @updatedAt
  userId String
  User   User   @relation
  type NotificationType
  Notifications NotificationEvent[]
  @@unique([userId, type])
}
```

## API Methods

### Exposed Service Methods (via HTTP)

#### queue_weekly_summary()
- **HTTP Endpoint**: `POST /queue_weekly_summary`
- **Purpose**: Triggers weekly summary generation for all active users
- **Process**: 
  1. Runs in background executor
  2. Queries users active in last 7 days
  3. Queues summary notification for each user
- **Used by**: Scheduler service (via cron)

#### process_existing_batches(notification_types: list[NotificationType])
- **HTTP Endpoint**: `POST /process_existing_batches`
- **Purpose**: Processes aged-out batches for specified notification types
- **Process**:
  1. Runs in background executor
  2. Retrieves all batches for given types
  3. Checks if oldest message exceeds max delay
  4. Sends batched email if aged out
  5. Clears processed batches
- **Used by**: Scheduler service (via cron)

#### discord_system_alert(content: str)
- **HTTP Endpoint**: `POST /discord_system_alert`
- **Purpose**: Sends system alerts to Discord channel
- **Async**: Yes (converted to sync by client)
- **Used by**: Monitoring services

### Direct Queue Functions (not via HTTP)

#### queue_notification(event: NotificationEventModel) -> NotificationResult
- **Purpose**: Queue a notification (sync version)
- **Used by**: ExecutionManager (same process)
- **Direct RabbitMQ**: Yes

#### queue_notification_async(event: NotificationEventModel) -> NotificationResult
- **Purpose**: Queue a notification (async version)
- **Used by**: Credit system (async context)
- **Direct RabbitMQ**: Yes

## Message Processing Flow

### 1. Message Routing
```python
def get_routing_key(event_type: NotificationType) -> str:
    strategy = NotificationTypeOverride(event_type).strategy
    if strategy == QueueType.IMMEDIATE:
        return f"notification.immediate.{event_type.value}"
    elif strategy == QueueType.BATCH:
        return f"notification.batch.{event_type.value}"
    # ... etc
```

### 2. Queue Processing Methods

#### _process_immediate(message: str) -> bool
1. Parse message to NotificationEventModel
2. Retrieve user email
3. Check user preferences and email verification
4. Send email immediately via EmailSender
5. Return True if successful

#### _process_batch(message: str) -> bool
1. Parse message to NotificationEventModel
2. Add to user's notification batch
3. Check if batch is old enough (based on delay)
4. If aged out:
   - Retrieve all batch messages
   - Send combined email
   - Clear batch
5. Return True if processed or batched

#### _process_summary(message: str) -> bool
1. Parse message to SummaryParamsEventModel
2. Gather summary data (credits, executions, etc.)
   - **Note**: Currently returns hardcoded placeholder data
3. Format and send summary email
4. Return True if successful

#### _process_admin_message(message: str) -> bool
1. Parse message
2. Send to configured admin email
3. No user preference checks
4. Return True if successful

## Email Delivery

### EmailSender Class

#### Template Loading
- Base template: `templates/base.html.jinja2`
- Notification templates: `templates/{notification_type}.html.jinja2`
- Subject templates from NotificationTypeOverride
- **Note**: Templates use `.html.jinja2` extension, not just `.html`

#### Email Composition
```python
def send_templated(
    notification: NotificationType,
    user_email: str,
    data: NotificationEventModel | list[NotificationEventModel],
    user_unsub_link: str | None = None
)
```

#### Postmark Integration
- API Token: `settings.secrets.postmark_server_api_token`
- Sender Email: `settings.config.postmark_sender_email`
- Headers:
  - `List-Unsubscribe-Post: List-Unsubscribe=One-Click`
  - `List-Unsubscribe: <{unsubscribe_link}>`

## User Preferences and Permissions

### Email Verification Check
```python
validated_email = get_db().get_user_email_verification(user_id)
```

### Notification Preferences
```python
preferences = get_db().get_user_notification_preference(user_id).preferences
# Returns dict[NotificationType, bool]
```

### Preference Fields in User Model
- `notifyOnAgentRun`
- `notifyOnZeroBalance`
- `notifyOnLowBalance`
- `notifyOnBlockExecutionFailed`
- `notifyOnContinuousAgentError`
- `notifyOnDailySummary`
- `notifyOnWeeklySummary`
- `notifyOnMonthlySummary`

### Unsubscribe Link Generation
```python
def generate_unsubscribe_link(user_id: str) -> str:
    # HMAC-SHA256 signed token
    # Format: base64(user_id:signature_hex)
    # URL: {platform_base_url}/api/email/unsubscribe?token={token}
```

## Batching Logic

### Batch Delays (get_batch_delay)

**Note**: The delay configuration exists for multiple notification types, but only notifications with `QueueType.BATCH` strategy actually use batching. Others use different strategies:

- `AGENT_RUN`: 1 day (Strategy: BATCH - actually uses batching)
- `ZERO_BALANCE`: 60 minutes configured (Strategy: BACKOFF - not batched)
- `LOW_BALANCE`: 60 minutes configured (Strategy: IMMEDIATE - sent immediately)
- `BLOCK_EXECUTION_FAILED`: 60 minutes configured (Strategy: BACKOFF - not batched)
- `CONTINUOUS_AGENT_ERROR`: 60 minutes configured (Strategy: BACKOFF - not batched)

### Batch Processing
1. Messages added to UserNotificationBatch
2. Oldest message timestamp tracked
3. When `oldest_timestamp + delay < now()`:
   - Batch is processed
   - All messages sent in single email
   - Batch cleared

## Service Lifecycle

### Startup
1. Initialize FastAPI app with exposed endpoints
2. Start HTTP server on port 8007
3. Initialize RabbitMQ connection
4. Create/verify exchanges and queues
5. Set up queue consumers
6. Start processing loop

### Main Loop
```python
while self.running:
    await self._run_queue(immediate_queue, self._process_immediate, ...)
    await self._run_queue(admin_queue, self._process_admin_message, ...)
    await self._run_queue(batch_queue, self._process_batch, ...)
    await self._run_queue(summary_queue, self._process_summary, ...)
    await asyncio.sleep(0.1)
```

### Shutdown
1. Set `running = False`
2. Disconnect RabbitMQ
3. Cleanup resources

## Configuration

### Environment Variables
```python
# Service Configuration
notification_service_port: int = 8007

# Email Configuration  
postmark_sender_email: str = "invalid@invalid.com"
refund_notification_email: str = "refund@agpt.co"

# Security
unsubscribe_secret_key: str = ""

# Secrets
postmark_server_api_token: str = ""
postmark_webhook_token: str = ""
discord_bot_token: str = ""

# Platform URLs
platform_base_url: str
frontend_base_url: str
```

## Error Handling

### Message Processing Errors
- Failed messages sent to dead letter queue
- Validation errors logged but don't crash service
- Connection errors trigger retry with `@continuous_retry()`

### RabbitMQ ACK/NACK Protocol
- Success: `message.ack()`
- Failure: `message.reject(requeue=False)`
- Timeout/Queue empty: Continue loop

### HTTP Endpoint Errors
- Wrapped in RemoteCallError for client
- Automatic retry available via client configuration
- Connection failures tracked and logged

## System Integrations

### DatabaseManagerClient
- User email retrieval
- Email verification status
- Notification preferences
- Batch management
- Active user queries

### Discord Integration
- Uses SendDiscordMessageBlock
- Configured via discord_bot_token
- For system alerts only

## Implementation Checklist

1. **Core Service**
   - [ ] AppService implementation with @expose decorators
   - [ ] FastAPI endpoint generation
   - [ ] RabbitMQ connection management
   - [ ] Queue consumer setup
   - [ ] Message routing logic

2. **Service Client**
   - [ ] NotificationManagerClient implementation
   - [ ] HTTP client configuration
   - [ ] Method mapping to service endpoints
   - [ ] Async-to-sync conversions

3. **Message Processing**
   - [ ] Parse and validate all notification types
   - [ ] Implement all queue strategies
   - [ ] Batch management with delays
   - [ ] Summary data gathering

4. **Email Delivery**
   - [ ] Postmark integration
   - [ ] Template loading and rendering
   - [ ] Unsubscribe header support
   - [ ] HTML email composition

5. **User Management**
   - [ ] Preference checking
   - [ ] Email verification
   - [ ] Unsubscribe link generation
   - [ ] Daily limit tracking

6. **Batching System**
   - [ ] Database batch operations
   - [ ] Age-out checking
   - [ ] Batch clearing after send
   - [ ] Oldest message tracking

7. **Error Handling**
   - [ ] Dead letter queue routing
   - [ ] Message rejection on failure
   - [ ] Continuous retry wrapper
   - [ ] Validation error logging

8. **Scheduled Operations**
   - [ ] Weekly summary generation
   - [ ] Batch processing triggers
   - [ ] Background executor usage

## Security Considerations

1. **Service-to-Service Communication**:
   - HTTP endpoints only accessible internally
   - No authentication on service endpoints (internal network only)
   - Service discovery via host/port configuration

2. **User Security**:
   - Email verification required for all user notifications
   - Unsubscribe tokens HMAC-signed
   - User preferences enforced

3. **Admin Notifications**:
   - Separate queue, no user preference checks
   - Fixed admin email configuration

## Testing Considerations

1. **Unit Tests**
   - Message parsing and validation
   - Routing key generation
   - Batch delay calculations
   - Template rendering

2. **Integration Tests**
   - HTTP endpoint accessibility
   - Service client method calls
   - RabbitMQ message flow
   - Database batch operations
   - Email sending (mock Postmark)

3. **Load Tests**
   - High volume message processing
   - Concurrent HTTP requests
   - Batch accumulation limits
   - Memory usage under load

## Implementation Status Notes

1. **Backoff Strategy**: While `QueueType.BACKOFF` is defined and used by several notification types (ZERO_BALANCE, BLOCK_EXECUTION_FAILED, CONTINUOUS_AGENT_ERROR), the actual exponential backoff processing logic is not implemented. These messages are routed to immediate queue.

2. **Summary Data**: The `_gather_summary_data()` method currently returns hardcoded placeholder values rather than querying actual execution data from the database.

3. **Batch Processing**: Only `AGENT_RUN` notifications actually use batch processing. Other notification types with configured delays use different strategies (IMMEDIATE or BACKOFF).

## Future Enhancements

1. **Additional Channels**
   - SMS notifications (not implemented)
   - Webhook notifications (not implemented)
   - In-app notifications

2. **Advanced Batching**
   - Dynamic batch sizes
   - Priority-based processing
   - Custom delay configurations

3. **Analytics**
   - Delivery tracking
   - Open/click rates
   - Notification effectiveness metrics

4. **Service Improvements**
   - Authentication for HTTP endpoints
   - Rate limiting per user
   - Circuit breaker patterns
   - Implement actual backoff processing for BACKOFF strategy
   - Implement real summary data gathering