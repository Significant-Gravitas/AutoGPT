# DatabaseManager Technical Specification

## Executive Summary

This document provides a complete technical specification for implementing a drop-in replacement for the AutoGPT Platform's DatabaseManager service. The replacement must maintain 100% API compatibility while preserving all functional behaviors, security requirements, and performance characteristics.

## 1. System Overview

### 1.1 Purpose
The DatabaseManager is a centralized service that provides database access for the AutoGPT Platform's executor system. It encapsulates all database operations behind a service interface, enabling distributed execution while maintaining data consistency and security.

### 1.2 Architecture Pattern
- **Service Type**: HTTP-based microservice using FastAPI
- **Communication**: RPC-style over HTTP with JSON serialization
- **Base Class**: Inherits from `AppService` (backend.util.service)
- **Client Classes**: `DatabaseManagerClient` (sync) and `DatabaseManagerAsyncClient` (async)
- **Port**: Configurable via `config.database_api_port`

### 1.3 Critical Requirements
1. **API Compatibility**: All 40+ exposed methods must maintain exact signatures
2. **Type Safety**: Full type preservation across service boundaries
3. **User Isolation**: All operations must respect user_id boundaries
4. **Transaction Support**: Maintain ACID properties for critical operations
5. **Event Publishing**: Maintain Redis event bus integration for real-time updates

## 2. Service Implementation Requirements

### 2.1 Base Service Class

```python
from backend.util.service import AppService, expose
from backend.util.settings import Config
from backend.data import db
import logging

class DatabaseManager(AppService):
    """
    REQUIRED: Inherit from AppService to get:
    - Automatic endpoint generation via @expose decorator
    - Built-in health checks at /health
    - Request/response serialization
    - Error handling and logging
    """
    
    def run_service(self) -> None:
        """REQUIRED: Initialize database connection before starting service"""
        logger.info(f"[{self.service_name}] ⏳ Connecting to Database...")
        self.run_and_wait(db.connect())  # CRITICAL: Must connect to database
        super().run_service()  # Start HTTP server
    
    def cleanup(self):
        """REQUIRED: Clean disconnect on shutdown"""
        super().cleanup()
        logger.info(f"[{self.service_name}] ⏳ Disconnecting Database...")
        self.run_and_wait(db.disconnect())  # CRITICAL: Must disconnect cleanly
    
    @classmethod
    def get_port(cls) -> int:
        """REQUIRED: Return configured port"""
        return config.database_api_port
```

### 2.2 Method Exposure Pattern

```python
@staticmethod
def _(f: Callable[P, R], name: str | None = None) -> Callable[Concatenate[object, P], R]:
    """
    REQUIRED: Helper to expose methods with proper signatures
    - Preserves function name for endpoint generation
    - Maintains type information
    - Adds 'self' parameter for instance binding
    """
    if name is not None:
        f.__name__ = name
    return cast(Callable[Concatenate[object, P], R], expose(f))
```

### 2.3 Database Connection Management

**REQUIRED: Use Prisma ORM with these exact configurations:**

```python
from prisma import Prisma

prisma = Prisma(
    auto_register=True,
    http={"timeout": HTTP_TIMEOUT},  # Default: 120 seconds
    datasource={"url": DATABASE_URL}
)

# Connection lifecycle
async def connect():
    await prisma.connect()

async def disconnect():
    await prisma.disconnect()
```

### 2.4 Transaction Support

**REQUIRED: Implement both regular and locked transactions:**

```python
async def transaction(timeout: float | None = None):
    """Regular database transaction"""
    async with prisma.tx(timeout=timeout) as tx:
        yield tx

async def locked_transaction(key: str, timeout: float | None = None):
    """Transaction with PostgreSQL advisory lock"""
    lock_key = zlib.crc32(key.encode("utf-8"))
    async with transaction(timeout=timeout) as tx:
        await tx.execute_raw("SELECT pg_advisory_xact_lock($1)", lock_key)
        yield tx
```

## 3. Complete API Specification

### 3.1 Execution Management APIs

#### get_graph_execution
```python
async def get_graph_execution(
    user_id: str,
    execution_id: str,
    *,
    include_node_executions: bool = False
) -> GraphExecution | GraphExecutionWithNodes | None
```
**Behavior**: 
- Returns execution only if user_id matches
- Optionally includes all node executions
- Returns None if not found or unauthorized

#### get_graph_executions
```python
async def get_graph_executions(
    user_id: str,
    graph_id: str | None = None,
    *,
    limit: int = 50,
    graph_version: int | None = None,
    cursor: str | None = None,
    preset_id: str | None = None
) -> tuple[list[GraphExecution], str | None]
```
**Behavior**:
- Paginated results with cursor
- Filter by graph_id, version, or preset_id
- Returns (executions, next_cursor)

#### create_graph_execution
```python
async def create_graph_execution(
    graph_id: str,
    graph_version: int,
    starting_nodes_input: dict[str, dict[str, Any]],
    user_id: str,
    preset_id: str | None = None
) -> GraphExecutionWithNodes
```
**Behavior**:
- Creates execution with status "QUEUED"
- Initializes all nodes with "PENDING" status
- Publishes creation event to Redis
- Uses locked transaction on graph_id

#### update_graph_execution_start_time
```python
async def update_graph_execution_start_time(
    graph_exec_id: str
) -> None
```
**Behavior**:
- Sets start_time to current timestamp
- Only updates if currently NULL

#### update_graph_execution_stats
```python
async def update_graph_execution_stats(
    graph_exec_id: str,
    status: AgentExecutionStatus | None = None,
    stats: dict[str, Any] | None = None
) -> GraphExecution | None
```
**Behavior**:
- Updates status and/or stats atomically
- Sets end_time if status is terminal (COMPLETED/FAILED)
- Publishes update event to Redis
- Returns updated execution

#### get_node_execution
```python
async def get_node_execution(
    node_exec_id: str
) -> NodeExecutionResult | None
```
**Behavior**:
- No user_id check (relies on graph execution security)
- Includes all input/output data

#### get_node_executions
```python
async def get_node_executions(
    graph_exec_id: str
) -> list[NodeExecutionResult]
```
**Behavior**:
- Returns all node executions for graph
- Ordered by creation time

#### get_latest_node_execution
```python
async def get_latest_node_execution(
    graph_exec_id: str,
    node_id: str
) -> NodeExecutionResult | None
```
**Behavior**:
- Returns most recent execution of specific node
- Used for retry/rerun scenarios

#### update_node_execution_status
```python
async def update_node_execution_status(
    node_exec_id: str,
    status: AgentExecutionStatus,
    execution_data: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None
) -> NodeExecutionResult
```
**Behavior**:
- Updates status atomically
- Sets end_time for terminal states
- Optionally updates stats/data
- Publishes event to Redis
- Returns updated execution

#### update_node_execution_status_batch
```python
async def update_node_execution_status_batch(
    execution_updates: list[NodeExecutionUpdate]
) -> list[NodeExecutionResult]
```
**Behavior**:
- Batch update multiple nodes in single transaction
- Each update can have different status/stats
- Publishes events for all updates
- Returns all updated executions

#### update_node_execution_stats
```python
async def update_node_execution_stats(
    node_exec_id: str,
    stats: dict[str, Any]
) -> NodeExecutionResult
```
**Behavior**:
- Updates only stats field
- Merges with existing stats
- Does not affect status

#### upsert_execution_input
```python
async def upsert_execution_input(
    node_id: str,
    graph_exec_id: str,
    input_name: str,
    input_data: Any,
    node_exec_id: str | None = None
) -> tuple[str, BlockInput]
```
**Behavior**:
- Creates or updates input data
- If node_exec_id not provided, creates node execution
- Serializes input_data to JSON
- Returns (node_exec_id, input_object)

#### upsert_execution_output
```python
async def upsert_execution_output(
    node_exec_id: str,
    output_name: str,
    output_data: Any
) -> None
```
**Behavior**:
- Creates or updates output data
- Serializes output_data to JSON
- No return value

#### get_execution_kv_data
```python
async def get_execution_kv_data(
    user_id: str,
    key: str
) -> Any | None
```
**Behavior**:
- User-scoped key-value storage
- Returns deserialized JSON data
- Returns None if key not found

#### set_execution_kv_data
```python
async def set_execution_kv_data(
    user_id: str,
    node_exec_id: str,
    key: str,
    data: Any
) -> Any | None
```
**Behavior**:
- Sets user-scoped key-value data
- Associates with node execution
- Serializes data to JSON
- Returns previous value or None

#### get_block_error_stats
```python
async def get_block_error_stats() -> list[BlockErrorStats]
```
**Behavior**:
- Aggregates error counts by block_id
- Last 7 days of data
- Groups by error type

### 3.2 Graph Management APIs

#### get_node
```python
async def get_node(
    node_id: str
) -> AgentNode | None
```
**Behavior**:
- Returns node with block data
- No user_id check (public blocks)

#### get_graph
```python
async def get_graph(
    graph_id: str,
    version: int | None = None,
    user_id: str | None = None,
    for_export: bool = False,
    include_subgraphs: bool = False
) -> GraphModel | None
```
**Behavior**:
- Returns latest version if version=None
- Checks user_id for private graphs
- for_export=True excludes internal fields
- include_subgraphs=True loads nested graphs

#### get_connected_output_nodes
```python
async def get_connected_output_nodes(
    node_id: str,
    output_name: str
) -> list[tuple[AgentNode, AgentNodeLink]]
```
**Behavior**:
- Returns downstream nodes connected to output
- Includes link metadata
- Used for execution flow

#### get_graph_metadata
```python
async def get_graph_metadata(
    graph_id: str,
    user_id: str
) -> GraphMetadata | None
```
**Behavior**:
- Returns graph metadata without full definition
- User must own or have access to graph

### 3.3 Credit System APIs

#### get_credits
```python
async def get_credits(
    user_id: str
) -> int
```
**Behavior**:
- Returns current credit balance
- Always non-negative

#### spend_credits
```python
async def spend_credits(
    user_id: str,
    cost: int,
    metadata: UsageTransactionMetadata
) -> int
```
**Behavior**:
- Deducts credits atomically
- Creates transaction record
- Throws InsufficientCredits if balance too low
- Returns new balance
- metadata includes: block_id, node_exec_id, context

### 3.4 User Management APIs

#### get_user_metadata
```python
async def get_user_metadata(
    user_id: str
) -> UserMetadata
```
**Behavior**:
- Returns user preferences and settings
- Creates default if not exists

#### update_user_metadata
```python
async def update_user_metadata(
    user_id: str,
    data: UserMetadataDTO
) -> UserMetadata
```
**Behavior**:
- Partial update of metadata
- Validates against schema
- Returns updated metadata

#### get_user_integrations
```python
async def get_user_integrations(
    user_id: str
) -> UserIntegrations
```
**Behavior**:
- Returns OAuth credentials
- Decrypts sensitive data
- Creates empty if not exists

#### update_user_integrations
```python
async def update_user_integrations(
    user_id: str,
    data: UserIntegrations
) -> None
```
**Behavior**:
- Updates integration credentials
- Encrypts sensitive data
- No return value

### 3.5 User Communication APIs

#### get_active_user_ids_in_timerange
```python
async def get_active_user_ids_in_timerange(
    start_time: datetime,
    end_time: datetime
) -> list[str]
```
**Behavior**:
- Returns users with graph executions in range
- Used for analytics/notifications

#### get_user_email_by_id
```python
async def get_user_email_by_id(
    user_id: str
) -> str | None
```
**Behavior**:
- Returns user's email address
- None if user not found

#### get_user_email_verification
```python
async def get_user_email_verification(
    user_id: str
) -> UserEmailVerification
```
**Behavior**:
- Returns email and verification status
- Used for notification filtering

#### get_user_notification_preference
```python
async def get_user_notification_preference(
    user_id: str
) -> NotificationPreference
```
**Behavior**:
- Returns notification settings
- Creates default if not exists

### 3.6 Notification APIs

#### create_or_add_to_user_notification_batch
```python
async def create_or_add_to_user_notification_batch(
    user_id: str,
    notification_type: NotificationType,
    notification_data: NotificationEvent
) -> UserNotificationBatchDTO
```
**Behavior**:
- Adds to existing batch or creates new
- Batches by type for efficiency
- Returns updated batch

#### empty_user_notification_batch
```python
async def empty_user_notification_batch(
    user_id: str,
    notification_type: NotificationType
) -> None
```
**Behavior**:
- Clears all notifications of type
- Used after sending batch

#### get_all_batches_by_type
```python
async def get_all_batches_by_type(
    notification_type: NotificationType
) -> list[UserNotificationBatchDTO]
```
**Behavior**:
- Returns all user batches of type
- Used by notification service

#### get_user_notification_batch
```python
async def get_user_notification_batch(
    user_id: str,
    notification_type: NotificationType
) -> UserNotificationBatchDTO | None
```
**Behavior**:
- Returns user's batch for type
- None if no batch exists

#### get_user_notification_oldest_message_in_batch
```python
async def get_user_notification_oldest_message_in_batch(
    user_id: str,
    notification_type: NotificationType
) -> NotificationEvent | None
```
**Behavior**:
- Returns oldest notification in batch
- Used for batch timing decisions

## 4. Client Implementation Requirements

### 4.1 Synchronous Client

```python
class DatabaseManagerClient(AppServiceClient):
    """
    REQUIRED: Synchronous client that:
    - Converts async methods to sync using endpoint_to_sync
    - Maintains exact method signatures
    - Handles connection pooling
    - Implements retry logic
    """
    
    @classmethod
    def get_service_type(cls):
        return DatabaseManager
    
    # Example method mapping
    get_graph_execution = endpoint_to_sync(DatabaseManager.get_graph_execution)
```

### 4.2 Asynchronous Client

```python
class DatabaseManagerAsyncClient(AppServiceClient):
    """
    REQUIRED: Async client that:
    - Directly references async methods
    - No conversion needed
    - Shares connection pool
    """
    
    @classmethod
    def get_service_type(cls):
        return DatabaseManager
    
    # Direct method reference
    get_graph_execution = DatabaseManager.get_graph_execution
```

## 5. Data Models

### 5.1 Core Enums

```python
class AgentExecutionStatus(str, Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"

class NotificationType(str, Enum):
    SYSTEM = "SYSTEM"
    REVIEW = "REVIEW"
    EXECUTION = "EXECUTION"
    MARKETING = "MARKETING"
```

### 5.2 Key Data Models

All models must exactly match the Prisma schema definitions. Key models include:

- `GraphExecution`: Execution metadata with stats
- `GraphExecutionWithNodes`: Includes all node executions
- `NodeExecutionResult`: Node execution with I/O data
- `GraphModel`: Complete graph definition
- `UserIntegrations`: OAuth credentials
- `UsageTransactionMetadata`: Credit usage context
- `NotificationEvent`: Individual notification data

## 6. Security Requirements

### 6.1 User Isolation
- **CRITICAL**: All user-scoped operations MUST filter by user_id
- Never expose data across user boundaries
- Use database-level row security where possible

### 6.2 Authentication
- Service assumes authentication handled by API gateway
- user_id parameter is trusted after authentication
- No additional auth checks within service

### 6.3 Data Protection
- Encrypt sensitive integration credentials
- Use HMAC for unsubscribe tokens
- Never log sensitive data

## 7. Performance Requirements

### 7.1 Connection Management
- Maintain persistent database connection
- Use connection pooling (default: 10 connections)
- Implement exponential backoff for retries

### 7.2 Query Optimization
- Use indexes for all WHERE clauses
- Batch operations where possible
- Limit default result sets (50 items)

### 7.3 Event Publishing
- Publish events asynchronously
- Don't block on event delivery
- Use fire-and-forget pattern

## 8. Error Handling

### 8.1 Standard Exceptions
```python
class InsufficientCredits(Exception):
    """Raised when user lacks credits"""

class NotFoundError(Exception):
    """Raised when entity not found"""

class AuthorizationError(Exception):
    """Raised when user lacks access"""
```

### 8.2 Error Response Format
```json
{
    "error": "error_type",
    "message": "Human readable message",
    "details": {}  // Optional additional context
}
```

## 9. Testing Requirements

### 9.1 Unit Tests
- Test each method in isolation
- Mock database calls
- Verify user_id filtering

### 9.2 Integration Tests
- Test with real database
- Verify transaction boundaries
- Test concurrent operations

### 9.3 Service Tests
- Test HTTP endpoint generation
- Verify serialization/deserialization
- Test error handling

## 10. Implementation Checklist

### Phase 1: Core Service Setup
- [ ] Create DatabaseManager class inheriting from AppService
- [ ] Implement run_service() with database connection
- [ ] Implement cleanup() with proper disconnect
- [ ] Configure port from settings
- [ ] Set up method exposure helper

### Phase 2: Execution APIs (15 methods)
- [ ] get_graph_execution
- [ ] get_graph_executions
- [ ] get_graph_execution_meta
- [ ] create_graph_execution
- [ ] update_graph_execution_start_time
- [ ] update_graph_execution_stats
- [ ] get_node_execution
- [ ] get_node_executions
- [ ] get_latest_node_execution
- [ ] update_node_execution_status
- [ ] update_node_execution_status_batch
- [ ] update_node_execution_stats
- [ ] upsert_execution_input
- [ ] upsert_execution_output
- [ ] get_execution_kv_data
- [ ] set_execution_kv_data
- [ ] get_block_error_stats

### Phase 3: Graph APIs (4 methods)
- [ ] get_node
- [ ] get_graph
- [ ] get_connected_output_nodes
- [ ] get_graph_metadata

### Phase 4: Credit APIs (2 methods)
- [ ] get_credits
- [ ] spend_credits

### Phase 5: User APIs (4 methods)
- [ ] get_user_metadata
- [ ] update_user_metadata
- [ ] get_user_integrations
- [ ] update_user_integrations

### Phase 6: Communication APIs (4 methods)
- [ ] get_active_user_ids_in_timerange
- [ ] get_user_email_by_id
- [ ] get_user_email_verification
- [ ] get_user_notification_preference

### Phase 7: Notification APIs (5 methods)
- [ ] create_or_add_to_user_notification_batch
- [ ] empty_user_notification_batch
- [ ] get_all_batches_by_type
- [ ] get_user_notification_batch
- [ ] get_user_notification_oldest_message_in_batch

### Phase 8: Client Implementation
- [ ] Create DatabaseManagerClient with sync methods
- [ ] Create DatabaseManagerAsyncClient with async methods
- [ ] Test client method generation
- [ ] Verify type preservation

### Phase 9: Integration Testing
- [ ] Test all methods with real database
- [ ] Verify user isolation
- [ ] Test error scenarios
- [ ] Performance testing
- [ ] Event publishing verification

### Phase 10: Deployment Validation
- [ ] Deploy to test environment
- [ ] Run integration test suite
- [ ] Verify backward compatibility
- [ ] Performance benchmarking
- [ ] Production deployment

## 11. Success Criteria

The implementation is successful when:

1. **All 40+ methods** produce identical outputs to the original
2. **Performance** is within 10% of original implementation
3. **All tests** pass without modification
4. **No breaking changes** to any client code
5. **Security boundaries** are maintained
6. **Event publishing** works identically
7. **Error handling** matches original behavior

## 12. Critical Implementation Notes

1. **DO NOT** modify any function signatures
2. **DO NOT** change any return types
3. **DO NOT** add new required parameters
4. **DO NOT** remove any functionality
5. **ALWAYS** maintain user_id isolation
6. **ALWAYS** publish events for state changes
7. **ALWAYS** use transactions for multi-step operations
8. **ALWAYS** handle errors exactly as original

This specification, when implemented correctly, will produce a drop-in replacement for the DatabaseManager that maintains 100% compatibility with the existing system.