# Graph Execution Stopping Mechanism - Key Files Reference

## Quick Navigation

### 1. REST API Endpoint (User Entry Point)
**File:** `/root/autogpt-1/autogpt_platform/backend/backend/server/routers/v1.py`
- **Lines 922-961:** Stop graph execution endpoint and logic
- **Function:** `stop_graph_run()` → `_stop_graph_run()`
- **Purpose:** HTTP endpoint to initiate execution stop
- **HTTP Method:** POST `/graphs/{graph_id}/executions/{graph_exec_id}/stop`

### 2. Core Stop Mechanism
**File:** `/root/autogpt-1/autogpt_platform/backend/backend/executor/utils.py`
- **Lines 615-681:** `stop_graph_execution()` function
- **Lines 551-608:** RabbitMQ queue configuration
- **Purpose:** Coordinates stop signal, publishes to message bus, polls for completion
- **Key Exchange:** `graph_execution_cancel` (FANOUT type)
- **Key Queue:** `graph_execution_cancel_queue`

### 3. AgentExecutorBlock (Child Agent Spawning)
**File:** `/root/autogpt-1/autogpt_platform/backend/backend/blocks/agent.py`
- **Lines 21-206:** Complete AgentExecutorBlock implementation
- **Lines 68-108:** `run()` method - creates child execution
- **Lines 110-183:** `_run()` method - listens to child execution events
- **Lines 185-206:** `_stop()` method - stops child execution
- **Purpose:** Executes sub-graphs/child agents within parent execution

### 4. Execution Manager (Worker Process)
**File:** `/root/autogpt-1/autogpt_platform/backend/backend/executor/manager.py`
- **Lines 122-137:** Worker initialization and graph execution entry point
- **Lines 622-680:** Main `on_graph_execution()` method
- **Lines 781-1053:** `_on_graph_execution()` - core execution loop with cancel detection
- **Lines 1055-1105:** `_cleanup_graph_execution()` - cleanup after stop
- **Lines 1415-1446:** `_handle_cancel_message()` - RabbitMQ cancel handler
- **Key Points:**
  - Lines 850, 927, 935: Cancel event checks (3 detection points)
  - Line 997: Final status determination
  - Line 1076: Task cancellation logic

### 5. Execution Status & Data Models
**File:** `/root/autogpt-1/autogpt_platform/backend/backend/data/execution.py`
- **Lines 91-118:** Status enum and valid state transitions
- **Lines 121-206:** GraphExecutionMeta and related models
- **Purpose:** Data structures for execution tracking
- **Key Statuses:** QUEUED, RUNNING, INCOMPLETE, COMPLETED, FAILED, TERMINATED

### 6. Database Schema
**File:** `/root/autogpt-1/autogpt_platform/backend/schema.prisma`
- **Lines 357-396:** AgentGraphExecution model (parent execution)
- **Lines 399-423:** AgentNodeExecution model (node executions)
- **Note:** No explicit `parentGraphExecutionId` field (limitation)
- **Key Indices:**
  - `@@index([userId, isDeleted, createdAt])` - for querying executions by user
  - `@@index([agentGraphExecutionId, agentNodeId, executionStatus])` - for graph execution nodes

### 7. Node Execution Progress Tracking
**File:** `/root/autogpt-1/autogpt_platform/backend/backend/executor/utils.py`
- **Lines 797-900:** NodeExecutionProgress class
- **Key Methods:**
  - `stop()` - cancels all running tasks
  - `wait_for_done()` - waits for cancellation to complete
  - `pop_output()` - retrieves execution output

### 8. Cancel Event Model
**File:** `/root/autogpt-1/autogpt_platform/backend/backend/executor/utils.py`
- **Lines 611-613:** CancelExecutionEvent Pydantic model
- **Purpose:** Message structure for RabbitMQ cancel events

---

## Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ User REST API Call (POST /stop)                         │
│ File: routers/v1.py:922                                 │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ _stop_graph_run() / stop_graph_execution()              │
│ File: executor/utils.py:615                             │
│ • Publish CancelExecutionEvent to RabbitMQ              │
│ • Poll for status changes                               │
│ • Force terminate if QUEUED/INCOMPLETE                  │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ RabbitMQ: graph_execution_cancel (FANOUT)               │
│ All listening executors receive the message             │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ ExecutionManager._handle_cancel_message()               │
│ File: executor/manager.py:1415                          │
│ • Set cancel_event threading.Event                      │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ _on_graph_execution() - Main Loop                       │
│ File: executor/manager.py:781                           │
│ • Check: if cancel.is_set() (lines 850, 927, 935)       │
│ • Break from execution loop                             │
│ • Determine status: TERMINATED                          │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ _cleanup_graph_execution()                              │
│ File: executor/manager.py:1055                          │
│ • Cancel all running node tasks                         │
│ • Wait for tasks to complete                            │
│ • Update DB with TERMINATED status                      │
│ • Clean up files                                        │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ Child Agent (if exists)                                 │
│ File: blocks/agent.py:110                               │
│ • If parent's exception caught                          │
│ • Call _stop() on child execution (line 100)            │
│ • Otherwise: Child continues (BUG!)                     │
└─────────────────────────────────────────────────────────┘
```

---

## Key Classes & Methods

### ExecutionManager
- `init_worker()` - Initialize worker process
- `execute_graph()` - Entry point for graph execution
- `on_graph_execution()` - Public method for execution
- `_on_graph_execution()` - Core execution loop
- `_handle_cancel_message()` - Cancel event handler
- `_cleanup_graph_execution()` - Cleanup after stop
- `_handle_run_message()` - RabbitMQ run message handler

### AgentExecutorBlock
- `run()` - Main entry point
- `_run()` - Event listening loop
- `_stop()` - Stop child execution

### NodeExecutionProgress
- `add_task()` - Add async task
- `stop()` - Cancel all tasks
- `wait_for_done()` - Wait for cancellation
- `pop_output()` - Retrieve output
- `is_done()` - Check if complete

---

## Important Constants

```python
# RabbitMQ Configuration
GRAPH_EXECUTION_EXCHANGE = "graph_execution" (DIRECT)
GRAPH_EXECUTION_QUEUE_NAME = "graph_execution_queue"
GRAPH_EXECUTION_ROUTING_KEY = "graph_execution.run"

GRAPH_EXECUTION_CANCEL_EXCHANGE = "graph_execution_cancel" (FANOUT)
GRAPH_EXECUTION_CANCEL_QUEUE_NAME = "graph_execution_cancel_queue"

GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 86400  # 1 day

# Default timeout for stop operation
wait_timeout = 15.0 seconds
```

---

## Critical Bug: Missing Cascading Stop

**Location:** `/root/autogpt-1/autogpt_platform/backend/backend/blocks/agent.py:68`

**Issue:** When parent execution is terminated:
1. Parent's execution loop breaks (line 850, 927, 935)
2. Parent's _cleanup_graph_execution() runs
3. Parent status set to TERMINATED
4. **BUT:** Child execution is NOT stopped

**Why:** No explicit parent-child relationship in database
- Line 72-78: Child created with independent graph_exec_id
- Line 128-132: Parent subscribes to child events
- **MISSING:** When parent receives cancel, it doesn't propagate to child

**Where It Should Be Fixed:**
1. Add parent_execution_id tracking to schema
2. Add cascading stop in stop_graph_execution()
3. Query for child executions and stop them
4. Or: Cancel event from parent should signal child

---

## Deployment Notes

### RabbitMQ Exchanges & Queues
- `graph_execution` exchange: Distributes run tasks via DIRECT routing
- `graph_execution_cancel` exchange: Broadcasts cancel to all workers (FANOUT)
- Both are durable and won't auto-delete

### Timeout Behavior
- **QUEUED/INCOMPLETE:** Force terminate immediately (no timeout)
- **RUNNING:** Poll with 100ms sleep, 15-second timeout
- **AgentExecutorBlock._stop():** 3600-second (1 hour) timeout on child
- **Cleanup:** 3600-second timeout for task cancellation

### Thread Safety
- Uses `threading.Event` for cancel signaling
- Uses `threading.Lock` for stats updates
- Uses `asyncio` for concurrent node execution
- Each worker process has its own thread pool

---

## Testing/Debugging Tips

### Check Active Executions
```sql
SELECT id, executionStatus, agentGraphId, createdAt 
FROM "AgentGraphExecution" 
WHERE userId = 'user_id' 
ORDER BY createdAt DESC;
```

### Check for Orphaned Child Executions
```sql
SELECT DISTINCT age.agentGraphId 
FROM "AgentGraphExecution" age
WHERE age.userId = 'user_id' 
AND age.executionStatus = 'RUNNING'
GROUP BY age.agentGraphId
HAVING COUNT(*) > 1;  -- Multiple executions of same graph
```

### Monitor RabbitMQ
```bash
# List queues
rabbitmqctl list_queues

# Monitor cancel messages
rabbitmqctl trace_on
```

### Enable Debug Logging
```python
# In executor/utils.py and executor/manager.py
logger.debug()  # Already present, check log level
```
