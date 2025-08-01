# AutoGPT Platform Scheduler Technical Specification

## Executive Summary

This document provides a comprehensive technical specification for the AutoGPT Platform Scheduler service. The scheduler is responsible for managing scheduled graph executions, system monitoring tasks, and periodic maintenance operations. This specification is designed to enable a complete reimplementation that maintains 100% compatibility with the existing system.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Service Implementation](#service-implementation)
3. [Data Models](#data-models)
4. [API Endpoints](#api-endpoints)
5. [Database Schema](#database-schema)
6. [External Dependencies](#external-dependencies)
7. [Authentication & Authorization](#authentication--authorization)
8. [Process Management](#process-management)
9. [Error Handling](#error-handling)
10. [Configuration](#configuration)
11. [Testing Strategy](#testing-strategy)

## System Architecture

### Overview

The scheduler operates as an independent microservice within the AutoGPT platform, implementing the `AppService` base class pattern. It runs on a dedicated port (default: 8003) and exposes HTTP/JSON-RPC endpoints for communication with other services.

### Core Components

1. **Scheduler Service** (`backend/executor/scheduler.py:156`)
   - Extends `AppService` base class
   - Manages APScheduler instance with multiple jobstores
   - Handles lifecycle management and graceful shutdown

2. **Scheduler Client** (`backend/executor/scheduler.py:354`)
   - Extends `AppServiceClient` base class
   - Provides async/sync method wrappers for RPC calls
   - Implements automatic retry and connection pooling

3. **Entry Points**
   - Main executable: `backend/scheduler.py`
   - Service launcher: `backend/app.py`

## Service Implementation

### Base Service Pattern

```python
class Scheduler(AppService):
    scheduler: BlockingScheduler
    
    def __init__(self, register_system_tasks: bool = True):
        self.register_system_tasks = register_system_tasks
    
    @classmethod
    def get_port(cls) -> int:
        return config.execution_scheduler_port  # Default: 8003
    
    @classmethod
    def db_pool_size(cls) -> int:
        return config.scheduler_db_pool_size  # Default: 3
    
    def run_service(self):
        # Initialize scheduler with jobstores
        # Register system tasks if enabled
        # Start scheduler blocking loop
    
    def cleanup(self):
        # Graceful shutdown of scheduler
        # Wait=False for immediate termination
```

### Jobstore Configuration

The scheduler uses three distinct jobstores:

1. **EXECUTION** (`Jobstores.EXECUTION.value`)
   - Type: SQLAlchemyJobStore
   - Table: `apscheduler_jobs`
   - Purpose: Graph execution schedules
   - Persistence: Required

2. **BATCHED_NOTIFICATIONS** (`Jobstores.BATCHED_NOTIFICATIONS.value`)
   - Type: SQLAlchemyJobStore
   - Table: `apscheduler_jobs_batched_notifications`
   - Purpose: Batched notification processing
   - Persistence: Required

3. **WEEKLY_NOTIFICATIONS** (`Jobstores.WEEKLY_NOTIFICATIONS.value`)
   - Type: MemoryJobStore
   - Purpose: Weekly summary notifications
   - Persistence: Not required

### System Tasks

When `register_system_tasks=True`, the following monitoring tasks are registered:

1. **Weekly Summary Processing**
   - Job ID: `process_weekly_summary`
   - Schedule: `0 * * * *` (hourly)
   - Function: `monitoring.process_weekly_summary`
   - Jobstore: WEEKLY_NOTIFICATIONS

2. **Late Execution Monitoring**
   - Job ID: `report_late_executions`
   - Schedule: Interval (config.execution_late_notification_threshold_secs)
   - Function: `monitoring.report_late_executions`
   - Jobstore: EXECUTION

3. **Block Error Rate Monitoring**
   - Job ID: `report_block_error_rates`
   - Schedule: Interval (config.block_error_rate_check_interval_secs)
   - Function: `monitoring.report_block_error_rates`
   - Jobstore: EXECUTION

4. **Cloud Storage Cleanup**
   - Job ID: `cleanup_expired_files`
   - Schedule: Interval (config.cloud_storage_cleanup_interval_hours * 3600)
   - Function: `cleanup_expired_files`
   - Jobstore: EXECUTION

## Data Models

### GraphExecutionJobArgs

```python
class GraphExecutionJobArgs(BaseModel):
    user_id: str
    graph_id: str
    graph_version: int
    cron: str
    input_data: BlockInput
    input_credentials: dict[str, CredentialsMetaInput] = Field(default_factory=dict)
```

### GraphExecutionJobInfo

```python
class GraphExecutionJobInfo(GraphExecutionJobArgs):
    id: str
    name: str
    next_run_time: str
    
    @staticmethod
    def from_db(job_args: GraphExecutionJobArgs, job_obj: JobObj) -> "GraphExecutionJobInfo":
        return GraphExecutionJobInfo(
            id=job_obj.id,
            name=job_obj.name,
            next_run_time=job_obj.next_run_time.isoformat(),
            **job_args.model_dump(),
        )
```

### NotificationJobArgs

```python
class NotificationJobArgs(BaseModel):
    notification_types: list[NotificationType]
    cron: str
```

### CredentialsMetaInput

```python
class CredentialsMetaInput(BaseModel, Generic[CP, CT]):
    id: str
    title: Optional[str] = None
    provider: CP
    type: CT
```

## API Endpoints

All endpoints are exposed via the `@expose` decorator and follow HTTP POST JSON-RPC pattern.

### 1. Add Graph Execution Schedule

**Endpoint**: `/add_graph_execution_schedule`

**Request Body**:
```json
{
    "user_id": "string",
    "graph_id": "string",
    "graph_version": "integer",
    "cron": "string (crontab format)",
    "input_data": {},
    "input_credentials": {},
    "name": "string (optional)"
}
```

**Response**: `GraphExecutionJobInfo`

**Behavior**:
- Creates APScheduler job with CronTrigger
- Uses job kwargs to store GraphExecutionJobArgs
- Sets `replace_existing=True` to allow updates
- Returns job info with generated ID and next run time

### 2. Delete Graph Execution Schedule

**Endpoint**: `/delete_graph_execution_schedule`

**Request Body**:
```json
{
    "schedule_id": "string",
    "user_id": "string"
}
```

**Response**: `GraphExecutionJobInfo`

**Behavior**:
- Validates schedule exists in EXECUTION jobstore
- Verifies user_id matches job's user_id
- Removes job from scheduler
- Returns deleted job info

**Errors**:
- `NotFoundError`: If job doesn't exist
- `NotAuthorizedError`: If user_id doesn't match

### 3. Get Graph Execution Schedules

**Endpoint**: `/get_graph_execution_schedules`

**Request Body**:
```json
{
    "graph_id": "string (optional)",
    "user_id": "string (optional)"
}
```

**Response**: `list[GraphExecutionJobInfo]`

**Behavior**:
- Retrieves all jobs from EXECUTION jobstore
- Filters by graph_id and/or user_id if provided
- Validates job kwargs as GraphExecutionJobArgs
- Skips invalid jobs (ValidationError)
- Only returns jobs with next_run_time set

### 4. System Task Endpoints

- `/execute_process_existing_batches` - Trigger batch processing
- `/execute_process_weekly_summary` - Trigger weekly summary
- `/execute_report_late_executions` - Trigger late execution report
- `/execute_report_block_error_rates` - Trigger error rate report
- `/execute_cleanup_expired_files` - Trigger file cleanup

### 5. Health Check

**Endpoints**: `/health_check`, `/health_check_async`
**Methods**: POST, GET
**Response**: "OK"

## Database Schema

### APScheduler Tables

The scheduler relies on APScheduler's SQLAlchemy jobstore schema:

1. **apscheduler_jobs**
   - id: VARCHAR (PRIMARY KEY)
   - next_run_time: FLOAT
   - job_state: BLOB/BYTEA (pickled job data)

2. **apscheduler_jobs_batched_notifications**
   - Same schema as above
   - Separate table for notification jobs

### Database Configuration

- URL extraction from `DIRECT_URL` environment variable
- Schema extraction from URL query parameter
- Connection pooling: `pool_size=db_pool_size()`, `max_overflow=0`
- Metadata schema binding for multi-schema support

## External Dependencies

### Required Services

1. **PostgreSQL Database**
   - Connection via `DIRECT_URL` environment variable
   - Schema support via URL parameter
   - APScheduler job persistence

2. **ExecutionManager** (via execution_utils)
   - Function: `add_graph_execution`
   - Called by: `execute_graph` job function
   - Purpose: Create graph execution entries

3. **NotificationManager** (via monitoring module)
   - Functions: `process_existing_batches`, `queue_weekly_summary`
   - Purpose: Notification processing

4. **Cloud Storage** (via util.cloud_storage)
   - Function: `cleanup_expired_files_async`
   - Purpose: File expiration management

### Python Dependencies

```
apscheduler>=3.10.0
sqlalchemy
pydantic>=2.0
httpx
uvicorn
fastapi
python-dotenv
tenacity
```

## Authentication & Authorization

### Service-Level Authentication

- No authentication required between internal services
- Services communicate via trusted internal network
- Host/port configuration via environment variables

### User-Level Authorization

- Authorization check in `delete_graph_execution_schedule`:
  - Validates `user_id` matches job's `user_id`
  - Raises `NotAuthorizedError` on mismatch
- No authorization for read operations (security consideration)

## Process Management

### Startup Sequence

1. Load environment variables via `dotenv.load_dotenv()`
2. Extract database URL and schema
3. Initialize BlockingScheduler with configured jobstores
4. Register system tasks (if enabled)
5. Add job execution listener
6. Start scheduler (blocking)

### Shutdown Sequence

1. Receive SIGTERM/SIGINT signal
2. Call `cleanup()` method
3. Shutdown scheduler with `wait=False`
4. Terminate process

### Multi-Process Architecture

- Runs as independent process via `AppProcess`
- Started by `run_processes()` in app.py
- Can run in foreground or background mode
- Automatic signal handling for graceful shutdown

## Error Handling

### Job Execution Errors

- Listener on `EVENT_JOB_ERROR` logs failures
- Errors in job functions are caught and logged
- Jobs continue to run on schedule despite failures

### RPC Communication Errors

- Automatic retry via `@conn_retry` decorator
- Configurable retry count and timeout
- Connection pooling with self-healing

### Database Connection Errors

- APScheduler handles reconnection automatically
- Pool exhaustion prevented by `max_overflow=0`
- Connection errors logged but don't crash service

## Configuration

### Environment Variables

- `DIRECT_URL`: PostgreSQL connection string (required)
- `{SERVICE_NAME}_HOST`: Override service host
- Standard logging configuration

### Config Settings (via Config class)

```python
execution_scheduler_port: int = 8003
scheduler_db_pool_size: int = 3
execution_late_notification_threshold_secs: int
block_error_rate_check_interval_secs: int
cloud_storage_cleanup_interval_hours: int
pyro_host: str = "localhost"
pyro_client_comm_timeout: float = 15
pyro_client_comm_retry: int = 3
rpc_client_call_timeout: int = 300
```

## Testing Strategy

### Unit Tests

1. Mock APScheduler for job management tests
2. Mock database connections
3. Test each RPC endpoint independently
4. Verify job serialization/deserialization

### Integration Tests

1. Test with real PostgreSQL instance
2. Verify job persistence across restarts
3. Test concurrent job execution
4. Validate cron expression parsing

### Critical Test Cases

1. **Job Persistence**: Jobs survive scheduler restart
2. **User Isolation**: Users can only delete their own jobs
3. **Concurrent Access**: Multiple clients can add/remove jobs
4. **Error Recovery**: Service recovers from database outages
5. **Resource Cleanup**: No memory/connection leaks

## Implementation Notes

### Key Design Decisions

1. **BlockingScheduler vs AsyncIOScheduler**: Uses BlockingScheduler for simplicity and compatibility with multiprocessing architecture

2. **Job Storage**: All job arguments stored in kwargs, not in job name/id

3. **Separate Jobstores**: Isolation between execution and notification jobs

4. **No Authentication**: Relies on network isolation for security

### Migration Considerations

1. APScheduler job format must be preserved exactly
2. Database schema cannot change without migration
3. RPC protocol must maintain compatibility
4. Environment variables must match existing deployment

### Performance Considerations

1. Database pool size limited to prevent exhaustion
2. No job result storage (fire-and-forget pattern)
3. Minimal logging in hot paths
4. Connection reuse via pooling

## Appendix: Critical Implementation Details

### Event Loop Management

```python
@thread_cached
def get_event_loop():
    return asyncio.new_event_loop()

def execute_graph(**kwargs):
    get_event_loop().run_until_complete(_execute_graph(**kwargs))
```

### Job Function Execution Context

- Jobs run in scheduler's process space
- Each job gets fresh event loop
- No shared state between job executions
- Exceptions logged but don't affect scheduler

### Cron Expression Format

- Uses standard crontab format via `CronTrigger.from_crontab()`
- Supports: minute hour day month day_of_week
- Special strings: @yearly, @monthly, @weekly, @daily, @hourly

This specification provides all necessary details to reimplement the scheduler service while maintaining 100% compatibility with the existing system. Any deviation from these specifications may result in system incompatibility.