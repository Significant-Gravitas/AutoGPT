# WebSocket API Technical Specification

## Overview

This document provides a complete technical specification for the AutoGPT Platform WebSocket API (`ws_api.py`). The WebSocket API provides real-time updates for graph and node execution events, enabling clients to monitor workflow execution progress.

## Architecture Overview

### Core Components

1. **WebSocket Server** (`ws_api.py`)
   - FastAPI application with WebSocket endpoint
   - Handles client connections and message routing
   - Authenticates clients via JWT tokens
   - Manages subscriptions to execution events

2. **Connection Manager** (`conn_manager.py`)
   - Maintains active WebSocket connections
   - Manages channel subscriptions
   - Routes execution events to subscribed clients
   - Handles connection lifecycle

3. **Event Broadcasting System**
   - Redis Pub/Sub based event bus
   - Asynchronous event broadcaster
   - Execution event propagation from backend services

## API Endpoint

### WebSocket Endpoint
- **URL**: `/ws`
- **Protocol**: WebSocket (ws:// or wss://)
- **Query Parameters**:
  - `token` (required when auth enabled): JWT authentication token

## Authentication

### JWT Token Authentication
- **When Required**: When `settings.config.enable_auth` is `True`
- **Token Location**: Query parameter `?token=<JWT_TOKEN>`
- **Token Validation**:
  ```python
  payload = parse_jwt_token(token)
  user_id = payload.get("sub")
  ```
- **JWT Requirements**:
  - Algorithm: Configured via `settings.JWT_ALGORITHM`
  - Secret Key: Configured via `settings.JWT_SECRET_KEY`
  - Audience: Must be "authenticated"
  - Claims: Must contain `sub` (user ID)

### Authentication Failures
- **4001**: Missing authentication token
- **4002**: Invalid token (missing user ID)
- **4003**: Invalid token (parsing error or expired)

### No-Auth Mode
- When `settings.config.enable_auth` is `False`
- Uses `DEFAULT_USER_ID` from `backend.data.user`

## Message Protocol

### Message Format
All messages use JSON format with the following structure:

```typescript
interface WSMessage {
  method: WSMethod;
  data?: Record<string, any> | any[] | string;
  success?: boolean;
  channel?: string;
  error?: string;
}
```

### Message Methods (WSMethod enum)

1. **Client-to-Server Methods**:
   - `SUBSCRIBE_GRAPH_EXEC`: Subscribe to specific graph execution
   - `SUBSCRIBE_GRAPH_EXECS`: Subscribe to all executions of a graph
   - `UNSUBSCRIBE`: Unsubscribe from a channel
   - `HEARTBEAT`: Keep-alive ping

2. **Server-to-Client Methods**:
   - `GRAPH_EXECUTION_EVENT`: Graph execution status update
   - `NODE_EXECUTION_EVENT`: Node execution status update
   - `ERROR`: Error message
   - `HEARTBEAT`: Keep-alive pong

## Subscription Models

### Subscribe to Specific Graph Execution
```typescript
interface WSSubscribeGraphExecutionRequest {
  graph_exec_id: string;
}
```
**Channel Key Format**: `{user_id}|graph_exec#{graph_exec_id}`

### Subscribe to All Graph Executions
```typescript
interface WSSubscribeGraphExecutionsRequest {
  graph_id: string;
}
```
**Channel Key Format**: `{user_id}|graph#{graph_id}|executions`

## Event Models

### Graph Execution Event
```typescript
interface GraphExecutionEvent {
  event_type: "graph_execution_update";
  id: string;                    // graph_exec_id
  user_id: string;
  graph_id: string;
  graph_version: number;
  preset_id?: string;
  status: ExecutionStatus;
  started_at: string;            // ISO datetime
  ended_at: string;              // ISO datetime
  inputs: Record<string, any>;
  outputs: Record<string, any>;
  stats?: {
    cost: number;                // cents
    duration: number;            // seconds
    duration_cpu_only: number;
    node_exec_time: number;
    node_exec_time_cpu_only: number;
    node_exec_count: number;
    node_error_count: number;
    error?: string;
  };
}
```

### Node Execution Event
```typescript
interface NodeExecutionEvent {
  event_type: "node_execution_update";
  user_id: string;
  graph_id: string;
  graph_version: number;
  graph_exec_id: string;
  node_exec_id: string;
  node_id: string;
  block_id: string;
  status: ExecutionStatus;
  input_data: Record<string, any>;
  output_data: Record<string, any>;
  add_time: string;              // ISO datetime
  queue_time?: string;           // ISO datetime
  start_time?: string;           // ISO datetime
  end_time?: string;             // ISO datetime
}
```

### Execution Status Enum
```typescript
enum ExecutionStatus {
  INCOMPLETE = "INCOMPLETE",
  QUEUED = "QUEUED",
  RUNNING = "RUNNING",
  COMPLETED = "COMPLETED",
  FAILED = "FAILED"
}
```

## Message Flow Examples

### 1. Subscribe to Graph Execution
```json
// Client → Server
{
  "method": "subscribe_graph_execution",
  "data": {
    "graph_exec_id": "exec-123"
  }
}

// Server → Client (Success)
{
  "method": "subscribe_graph_execution",
  "success": true,
  "channel": "user-456|graph_exec#exec-123"
}
```

### 2. Receive Execution Updates
```json
// Server → Client (Graph Update)
{
  "method": "graph_execution_event",
  "channel": "user-456|graph_exec#exec-123",
  "data": {
    "event_type": "graph_execution_update",
    "id": "exec-123",
    "user_id": "user-456",
    "graph_id": "graph-789",
    "status": "RUNNING",
    // ... other fields
  }
}

// Server → Client (Node Update)
{
  "method": "node_execution_event",
  "channel": "user-456|graph_exec#exec-123",
  "data": {
    "event_type": "node_execution_update",
    "node_exec_id": "node-exec-111",
    "status": "COMPLETED",
    // ... other fields
  }
}
```

### 3. Heartbeat
```json
// Client → Server
{
  "method": "heartbeat",
  "data": "ping"
}

// Server → Client
{
  "method": "heartbeat",
  "data": "pong",
  "success": true
}
```

### 4. Error Handling
```json
// Server → Client (Invalid Message)
{
  "method": "error",
  "success": false,
  "error": "Invalid message format. Review the schema and retry"
}
```

## Event Broadcasting Architecture

### Redis Pub/Sub Integration
1. **Event Bus Name**: Configured via `config.execution_event_bus_name`
2. **Channel Pattern**: `{event_bus_name}/{channel_key}`
3. **Event Flow**:
   - Execution services publish events to Redis
   - Event broadcaster listens to Redis pattern `*`
   - Events are routed to WebSocket connections based on subscriptions

### Event Broadcaster
- Runs as continuous async task using `@continuous_retry()` decorator
- Listens to all execution events via `AsyncRedisExecutionEventBus`
- Calls `ConnectionManager.send_execution_update()` for each event

## Connection Lifecycle

### Connection Establishment
1. Client connects to `/ws` endpoint
2. Authentication performed (JWT validation)
3. WebSocket accepted via `manager.connect_socket()`
4. Connection added to active connections set

### Message Processing Loop
1. Receive text message from client
2. Parse and validate as `WSMessage`
3. Route to appropriate handler based on `method`
4. Send response or error back to client

### Connection Termination
1. `WebSocketDisconnect` exception caught
2. `manager.disconnect_socket()` called
3. Connection removed from active connections
4. All subscriptions for that connection removed

## Error Handling

### Validation Errors
- **Invalid Message Format**: Returns error with method "error"
- **Invalid Message Data**: Returns error with specific validation message
- **Unknown Message Type**: Returns error indicating unsupported method

### Connection Errors
- WebSocket disconnections handled gracefully
- Failed event parsing logged but doesn't crash connection
- Handler exceptions logged and connection continues

## Configuration

### Environment Variables
```python
# WebSocket Server Configuration
websocket_server_host: str = "0.0.0.0"
websocket_server_port: int = 8001

# Authentication
enable_auth: bool = True

# CORS
backend_cors_allow_origins: List[str] = []

# Redis Event Bus
execution_event_bus_name: str = "autogpt:execution_event_bus"

# Message Size Limits
max_message_size_limit: int = 512000  # 512KB
```

### Security Headers
- CORS middleware applied with configured origins
- Credentials allowed for authenticated requests
- All methods and headers allowed (configurable)

## Deployment Requirements

### Dependencies
1. **FastAPI**: Web framework with WebSocket support
2. **Redis**: For pub/sub event broadcasting
3. **JWT Libraries**: For token validation
4. **Prisma**: Database ORM (for future graph access validation)

### Process Management
- Implements `AppProcess` interface for service lifecycle
- Runs via `uvicorn` ASGI server
- Graceful shutdown handling in `cleanup()` method

### Concurrent Connections
- No hard limit on WebSocket connections
- Memory usage scales with active connections
- Each connection maintains subscription set

## Implementation Checklist

To implement a compatible WebSocket API:

1. **Authentication**
   - [ ] JWT token validation from query parameters
   - [ ] Support for no-auth mode with default user ID
   - [ ] Proper error codes for auth failures

2. **Message Handling**
   - [ ] Parse and validate WSMessage format
   - [ ] Implement all client-to-server methods
   - [ ] Support all server-to-client event types
   - [ ] Proper error responses for invalid messages

3. **Subscription Management**
   - [ ] Channel key generation matching exact format
   - [ ] Support for both execution and graph-level subscriptions
   - [ ] Unsubscribe functionality
   - [ ] Clean up subscriptions on disconnect

4. **Event Broadcasting**
   - [ ] Listen to Redis pub/sub for execution events
   - [ ] Route events to correct subscribed connections
   - [ ] Handle both graph and node execution events
   - [ ] Maintain event order and completeness

5. **Connection Management**
   - [ ] Track active WebSocket connections
   - [ ] Handle graceful disconnections
   - [ ] Implement heartbeat/keepalive
   - [ ] Memory-efficient subscription storage

6. **Configuration**
   - [ ] Support all environment variables
   - [ ] CORS configuration for allowed origins
   - [ ] Configurable host/port binding
   - [ ] Redis connection configuration

7. **Error Handling**
   - [ ] Graceful handling of malformed messages
   - [ ] Logging of errors without dropping connections
   - [ ] Specific error messages for debugging
   - [ ] Recovery from Redis connection issues

## Testing Considerations

1. **Unit Tests**
   - Message parsing and validation
   - Channel key generation
   - Subscription management logic

2. **Integration Tests**
   - Full WebSocket connection flow
   - Event broadcasting from Redis
   - Multi-client subscription scenarios
   - Authentication success/failure cases

3. **Load Tests**
   - Many concurrent connections
   - High-frequency event broadcasting
   - Memory usage under load
   - Connection/disconnection cycles

## Security Considerations

1. **Authentication**: JWT tokens transmitted via query parameters (consider upgrading to headers)
2. **Authorization**: Currently no graph-level access validation (commented out in code)
3. **Rate Limiting**: No rate limiting implemented
4. **Message Size**: Limited by `max_message_size_limit` configuration
5. **Input Validation**: All inputs validated via Pydantic models

## Future Enhancements (Currently Commented Out)

1. **Graph Access Validation**: Verify user has read access to subscribed graphs
2. **Message Compression**: For large execution payloads
3. **Batch Updates**: Aggregate multiple events in single message
4. **Selective Field Subscription**: Subscribe to specific fields only