# Graph Execution Stopping Mechanism Analysis

## Executive Summary

The AutoGPT platform has a sophisticated graph execution system where:
1. Parent agents can spawn child agents via **AgentExecutorBlock**
2. Execution stopping is triggered via a **REST API endpoint** and **RabbitMQ message bus**
3. **Cascading stops** rely on listening to child execution completion events
4. **Current limitation**: No explicit parent-child tracking in the database schema

---

## 1. STOP GRAPH EXECUTION ENDPOINT & HANDLER

### Location
**File:** `/root/autogpt-1/autogpt_platform/backend/backend/server/routers/v1.py`
**Lines:** 922-961

### Endpoint Definition
```python
@v1_router.post(
    path="/graphs/{graph_id}/executions/{graph_exec_id}/stop",
    summary="Stop graph execution",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
async def stop_graph_run(
    graph_id: str, 
    graph_exec_id: str, 
    user_id: Annotated[str, Security(get_user_id)]
) -> execution_db.GraphExecutionMeta | None:
    res = await _stop_graph_run(...)
    return res[0] if res else None
```

### Stop Logic
The `_stop_graph_run` function:
1. **Fetches all executions** for a given graph in incomplete states
2. **Calls `stop_graph_execution()`** for each one
3. **Handles states**: `QUEUED`, `INCOMPLETE`, `RUNNING`

```python
async def _stop_graph_run(
    user_id: str,
    graph_id: Optional[str] = None,
    graph_exec_id: Optional[str] = None,
) -> list[execution_db.GraphExecutionMeta]:
    graph_execs = await execution_db.get_graph_executions(
        user_id=user_id,
        graph_id=graph_id,
        graph_exec_id=graph_exec_id,
        statuses=[
            execution_db.ExecutionStatus.INCOMPLETE,
            execution_db.ExecutionStatus.QUEUED,
            execution_db.ExecutionStatus.RUNNING,
        ],
    )
    stopped_execs = [
        execution_utils.stop_graph_execution(graph_exec_id=exec.id, user_id=user_id)
        for exec in graph_execs
    ]
    await asyncio.gather(*stopped_execs)
    return graph_execs
```

---

## 2. CORE STOP EXECUTION FUNCTION

### Location
**File:** `/root/autogpt-1/autogpt_platform/backend/backend/executor/utils.py`
**Lines:** 615-681

### Mechanism
The function implements a 3-step process:

```python
async def stop_graph_execution(
    user_id: str,
    graph_exec_id: str,
    wait_timeout: float = 15.0,
):
    """
    Mechanism:
    1. Set the cancel event
    2. Graph executor's cancel handler thread detects the event, terminates workers,
       reinitializes worker pool, and returns.
    3. Update execution statuses in DB and set `error` outputs to `"TERMINATED"`.
    """
```

#### Step 1: Publish Cancel Event via RabbitMQ
```python
queue_client = await get_async_execution_queue()
db = execution_db if prisma.is_connected() else get_database_manager_async_client()
await queue_client.publish_message(
    routing_key="",
    message=CancelExecutionEvent(graph_exec_id=graph_exec_id).model_dump_json(),
    exchange=GRAPH_EXECUTION_CANCEL_EXCHANGE,
)
```

**Exchange Configuration:**
- **Type:** FANOUT (broadcasts to all listening executors)
- **Queue:** `graph_execution_cancel_queue`
- **Purpose:** Notify active executor service about cancellation

#### Step 2: Wait for Execution to Stop (Polling Loop)
The function enters a timeout loop (default: 15 seconds) checking execution status:

```python
start_time = time.time()
while time.time() - start_time < wait_timeout:
    graph_exec = await db.get_graph_execution_meta(
        execution_id=graph_exec_id, user_id=user_id
    )
    
    # Case 1: Already terminal - success
    if graph_exec.status in [
        ExecutionStatus.TERMINATED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
    ]:
        await get_async_execution_event_bus().publish(graph_exec)
        return
    
    # Case 2: Still queued/incomplete - force terminate
    if graph_exec.status in [
        ExecutionStatus.QUEUED,
        ExecutionStatus.INCOMPLETE,
    ]:
        graph_exec.status = ExecutionStatus.TERMINATED
        await asyncio.gather(
            db.update_graph_execution_stats(
                graph_exec_id=graph_exec.id,
                status=ExecutionStatus.TERMINATED,
            ),
            get_async_execution_event_bus().publish(graph_exec),
        )
        return
    
    # Case 3: Still running - wait and retry
    if graph_exec.status == ExecutionStatus.RUNNING:
        await asyncio.sleep(0.1)

# Timeout failure
raise TimeoutError(...)
```

---

## 3. AGENT EXECUTOR BLOCK IMPLEMENTATION

### Location
**File:** `/root/autogpt-1/autogpt_platform/backend/backend/blocks/agent.py`
**Lines:** 21-206

### Class Definition
```python
class AgentExecutorBlock(Block):
    class Input(BlockSchema):
        user_id: str
        graph_id: str
        graph_version: int
        agent_name: Optional[str]
        inputs: BlockInput
        input_schema: dict
        output_schema: dict
        nodes_input_masks: Optional[NodesInputMasks]
    
    class Output(BlockSchema):
        pass  # Dynamic output schema from child graph
```

### Execution Flow: `run()` method

```python
async def run(self, input_data: Input, **kwargs) -> BlockOutput:
    from backend.executor import utils as execution_utils
    
    # STEP 1: Create child graph execution
    graph_exec = await execution_utils.add_graph_execution(
        graph_id=input_data.graph_id,
        graph_version=input_data.graph_version,
        user_id=input_data.user_id,
        inputs=input_data.inputs,
        nodes_input_masks=input_data.nodes_input_masks,
    )
    
    logger = execution_utils.LogMetadata(...)
    
    try:
        # STEP 2: Listen for execution events until completion
        async for name, data in self._run(
            graph_id=input_data.graph_id,
            graph_version=input_data.graph_version,
            graph_exec_id=graph_exec.id,
            user_id=input_data.user_id,
            logger=logger,
        ):
            yield name, data
    except BaseException as e:
        # STEP 3: On error, stop the child execution
        await self._stop(
            graph_exec_id=graph_exec.id,
            user_id=input_data.user_id,
            logger=logger,
        )
        logger.warning(...)
        raise
```

### Child Execution Event Listening: `_run()` method

```python
async def _run(
    self,
    graph_id: str,
    graph_version: int,
    graph_exec_id: str,
    user_id: str,
    logger,
) -> BlockOutput:
    from backend.data.execution import ExecutionEventType
    from backend.executor import utils as execution_utils
    
    event_bus = execution_utils.get_async_execution_event_bus()
    
    # Listen to execution events
    async for event in event_bus.listen(
        user_id=user_id,
        graph_id=graph_id,
        graph_exec_id=graph_exec_id,
    ):
        # Only process terminal events
        if event.status not in [
            ExecutionStatus.COMPLETED,
            ExecutionStatus.TERMINATED,
            ExecutionStatus.FAILED,
        ]:
            logger.debug(...)
            continue
        
        # Stop listening when graph execution completes
        if event.event_type == ExecutionEventType.GRAPH_EXEC_UPDATE:
            self.merge_stats(NodeExecutionStats(...))
            break
        
        # Extract and yield outputs from OUTPUT blocks
        if event.block_id:
            block = get_block(event.block_id)
            if block and block.block_type == BlockType.OUTPUT:
                output_name = event.input_data.get("name")
                for output_data in event.output_data.get("output", []):
                    yield output_name, output_data
```

### Stop Child Execution: `_stop()` method

```python
@func_retry
async def _stop(
    self,
    graph_exec_id: str,
    user_id: str,
    logger,
) -> None:
    from backend.executor import utils as execution_utils
    
    log_id = f"Graph exec-id: {graph_exec_id}"
    logger.info(f"Stopping execution of {log_id}")
    
    try:
        await execution_utils.stop_graph_execution(
            graph_exec_id=graph_exec_id,
            user_id=user_id,
            wait_timeout=3600,  # 1 hour timeout
        )
        logger.info(f"Execution {log_id} stopped successfully.")
    except TimeoutError as e:
        logger.error(f"Execution {log_id} stop timed out: {e}")
```

---

## 4. EXECUTION MANAGER - CANCEL HANDLER

### Location
**File:** `/root/autogpt-1/autogpt_platform/backend/backend/executor/manager.py`
**Lines:** 1415-1446

### Cancel Message Handler

```python
def _handle_cancel_message(
    self,
    _channel: BlockingChannel,
    _method: Basic.Deliver,
    _properties: BasicProperties,
    body: bytes,
):
    """
    Called whenever we receive a CANCEL message from the queue.
    (With auto_ack=True, message is considered 'acked' automatically.)
    """
    request = CancelExecutionEvent.model_validate_json(body)
    graph_exec_id = request.graph_exec_id
    
    if not graph_exec_id:
        logger.warning("Cancel message missing 'graph_exec_id'")
        return
    
    if graph_exec_id not in self.active_graph_runs:
        logger.debug(f"Cancel received for {graph_exec_id} but not active.")
        return
    
    # Get the cancel event for this execution
    _, cancel_event = self.active_graph_runs[graph_exec_id]
    logger.info(f"Received cancel for {graph_exec_id}")
    
    if not cancel_event.is_set():
        cancel_event.set()  # Signal the execution loop to stop
    else:
        logger.debug(f"Cancel already set for {graph_exec_id}")
```

### Active Runs Tracking
```python
self.active_graph_runs: dict[str, tuple[Future, threading.Event]] = {}
```

Each entry maps:
- `graph_exec_id` -> `(execution_future, cancel_event)`

The `cancel_event` is a threading.Event that signals the main execution loop to stop.

---

## 5. EXECUTION LOOP - CANCEL DETECTION

### Location
**File:** `/root/autogpt-1/autogpt_platform/backend/backend/executor/manager.py`
**Lines:** 781-1053

### Main Execution Loop with Cancel Polling

```python
def _on_graph_execution(
    self,
    graph_exec: GraphExecutionEntry,
    cancel: threading.Event,  # Cancel event from active_graph_runs
    log_metadata: LogMetadata,
    execution_stats: GraphExecutionStats,
    cluster_lock: ClusterLock,
) -> ExecutionStatus:
    execution_status: ExecutionStatus = ExecutionStatus.RUNNING
    error: Exception | None = None
    db_client = get_db_client()
    execution_stats_lock = threading.Lock()
    
    running_node_execution: dict[str, NodeExecutionProgress] = defaultdict(
        NodeExecutionProgress
    )
    
    try:
        # Pre-populate queue from DB (for resuming executions)
        for node_exec in db_client.get_node_executions(
            graph_exec.graph_exec_id,
            statuses=[
                ExecutionStatus.RUNNING,
                ExecutionStatus.QUEUED,
                ExecutionStatus.TERMINATED,
            ],
        ):
            node_entry = node_exec.to_node_execution_entry(graph_exec.user_context)
            execution_queue.add(node_entry)
        
        # ===== MAIN DISPATCH LOOP =====
        while not execution_queue.empty():
            # CHECK 1: Detect cancel signal
            if cancel.is_set():
                break  # Exit main loop
            
            queued_node_exec = execution_queue.get()
            
            # Charge usage cost...
            # Dispatch node execution...
            # Add to running_node_execution...
            
            # ===== POLLING LOOP =====
            while execution_queue.empty() and (
                running_node_execution or running_node_evaluation
            ):
                # CHECK 2: Detect cancel signal in polling loop
                if cancel.is_set():
                    break
                
                # Handle inflight node executions...
                for node_id, inflight_exec in list(running_node_execution.items()):
                    # CHECK 3: Check cancel again before processing
                    if cancel.is_set():
                        break
                    
                    # Process node outputs...
                    if output := inflight_exec.pop_output():
                        # Enqueue next nodes...
        
        # ===== FINAL STATUS DETERMINATION =====
        if cancel.is_set():
            execution_status = ExecutionStatus.TERMINATED
        elif error is not None:
            execution_status = ExecutionStatus.FAILED
        else:
            execution_status = ExecutionStatus.COMPLETED
        
        return execution_status
        
    finally:
        # Cleanup all running tasks
        self._cleanup_graph_execution(...)
```

### Cleanup Process
```python
def _cleanup_graph_execution(
    self,
    execution_queue: ExecutionQueue[NodeExecutionEntry],
    running_node_execution: dict[str, "NodeExecutionProgress"],
    running_node_evaluation: dict[str, Future],
    execution_status: ExecutionStatus,
    error: Exception | None,
    graph_exec_id: str,
    log_metadata: LogMetadata,
    db_client: "DatabaseManagerClient",
) -> None:
    # Cancel all running node execution tasks
    for node_id, inflight_exec in running_node_execution.items():
        if inflight_exec.is_done():
            continue
        log_metadata.info(f"Stopping node execution {node_id}")
        inflight_exec.stop()  # Cancel async tasks
    
    # Wait for all tasks to complete
    for node_id, inflight_exec in running_node_execution.items():
        try:
            inflight_exec.wait_for_done(timeout=3600.0)
        except TimeoutError:
            log_metadata.exception(
                f"Node execution #{node_id} did not stop in time..."
            )
    
    # Wait for evaluation futures
    for node_id, inflight_eval in running_node_evaluation.items():
        try:
            inflight_eval.result(timeout=3600.0)
        except TimeoutError:
            log_metadata.exception(...)
    
    # Update all queued nodes to TERMINATED status
    while queued_execution := execution_queue.get_or_none():
        update_node_execution_status(
            db_client=db_client,
            exec_id=queued_execution.node_exec_id,
            status=execution_status,
            stats={"error": str(error)} if error else None,
        )
    
    clean_exec_files(graph_exec_id)
```

---

## 6. EXECUTION STATUS MANAGEMENT

### Execution Status Enum
**Location:** `/root/autogpt-1/autogpt_platform/backend/backend/data/execution.py`

```python
ExecutionStatus = AgentExecutionStatus  # From Prisma
# Possible values: QUEUED, INCOMPLETE, RUNNING, COMPLETED, FAILED, TERMINATED
```

### Status Transitions
```python
VALID_STATUS_TRANSITIONS = {
    ExecutionStatus.QUEUED: [ExecutionStatus.INCOMPLETE],
    ExecutionStatus.RUNNING: [
        ExecutionStatus.INCOMPLETE,
        ExecutionStatus.QUEUED,
        ExecutionStatus.TERMINATED,  # For resuming halted execution
    ],
    ExecutionStatus.COMPLETED: [ExecutionStatus.RUNNING],
    ExecutionStatus.FAILED: [
        ExecutionStatus.INCOMPLETE,
        ExecutionStatus.QUEUED,
        ExecutionStatus.RUNNING,
    ],
    ExecutionStatus.TERMINATED: [
        ExecutionStatus.INCOMPLETE,
        ExecutionStatus.QUEUED,
        ExecutionStatus.RUNNING,
    ],
}
```

### Status Flow: Queued vs Running vs Stopped

```
Creation: INCOMPLETE/QUEUED
    ↓
Running: RUNNING (when executor picks it up)
    ↓
Terminal: COMPLETED/FAILED/TERMINATED
```

**Key States:**
- **QUEUED**: Waiting in RabbitMQ queue, not yet running
- **RUNNING**: Currently executing on a worker thread
- **INCOMPLETE**: Waiting for input data (dynamic graph)
- **TERMINATED**: Cancelled by user
- **COMPLETED**: Finished successfully
- **FAILED**: Encountered an error

---

## 7. DATABASE SCHEMA - EXECUTION RELATIONSHIPS

### Location
**File:** `/root/autogpt-1/autogpt_platform/backend/schema.prisma`
**Lines:** 357-423

### AgentGraphExecution Model
```prisma
model AgentGraphExecution {
  id        String    @id @default(uuid())
  createdAt DateTime  @default(now())
  updatedAt DateTime? @updatedAt
  startedAt DateTime?

  isDeleted Boolean @default(false)

  executionStatus AgentExecutionStatus @default(COMPLETED)

  agentGraphId      String
  agentGraphVersion Int
  AgentGraph        AgentGraph @relation(fields: [agentGraphId, agentGraphVersion], references: [id, version], onDelete: Cascade)

  agentPresetId String?
  AgentPreset   AgentPreset? @relation(fields: [agentPresetId], references: [id])

  inputs           Json?
  credentialInputs Json?
  nodesInputMasks  Json?

  NodeExecutions AgentNodeExecution[]

  userId String
  User   User   @relation(fields: [userId], references: [id], onDelete: Cascade)

  stats Json?

  isShared   Boolean   @default(false)
  shareToken String?   @unique
  sharedAt   DateTime?

  @@index([agentGraphId, agentGraphVersion])
  @@index([userId, isDeleted, createdAt])
  @@index([createdAt])
  @@index([agentPresetId])
  @@index([shareToken])
}
```

### AgentNodeExecution Model
```prisma
model AgentNodeExecution {
  id String @id @default(uuid())

  agentGraphExecutionId String
  GraphExecution        AgentGraphExecution @relation(fields: [agentGraphExecutionId], references: [id], onDelete: Cascade)

  agentNodeId String
  Node        AgentNode @relation(fields: [agentNodeId], references: [id], onDelete: Cascade)

  Input  AgentNodeExecutionInputOutput[] @relation("AgentNodeExecutionInput")
  Output AgentNodeExecutionInputOutput[] @relation("AgentNodeExecutionOutput")

  executionStatus AgentExecutionStatus @default(COMPLETED)
  executionData   Json?
  addedTime       DateTime             @default(now())
  queuedTime      DateTime?
  startedTime     DateTime?
  endedTime       DateTime?

  stats Json?

  @@index([agentGraphExecutionId, agentNodeId, executionStatus])
  @@index([agentNodeId, executionStatus])
  @@index([addedTime, queuedTime])
}
```

### Key Observations:

1. **No Parent Execution Reference**: The schema does NOT have a `parentGraphExecutionId` field
2. **Implicit Parent-Child**: Parent-child relationships are implicit through:
   - Parent creates child via `AgentExecutorBlock.run()`
   - Child's `agentGraphId` references a sub-graph
   - Timing allows inference (child creation time < parent execution time)
3. **Cascade Deletion**: `onDelete: Cascade` means deleting parent graph execution deletes all child node executions

---

## 8. HOW SUB-AGENTS ARE SPAWNED AND TRACKED

### Spawning Flow

```
Parent Agent Execution
    ↓
[AgentExecutorBlock Node]
    ↓
execute_node(...) called
    ↓
AgentExecutorBlock.run(input_data, **kwargs)
    ↓
await execution_utils.add_graph_execution(...)
    ↓
GraphExecutionEntry created
    ↓
Published to RabbitMQ execution queue
    ↓
Status: QUEUED
```

### Tracking Method: Event Bus Listening

Unlike explicit parent-child database relationships, tracking is done via:

1. **Event Bus Subscription**:
   - Parent (AgentExecutorBlock) calls `event_bus.listen(user_id, graph_id, graph_exec_id)`
   - Subscribes to execution events for specific child graph

2. **Execution Events**:
   - Child execution publishes events when nodes complete
   - When child reaches COMPLETED/TERMINATED/FAILED, event is published
   - Parent listening loop receives `ExecutionEvent` with status update
   - Parent breaks loop and yields outputs

3. **Implicit Tracking**:
   - Child execution ID is known (returned from `add_graph_execution()`)
   - Used to query event bus for specific child
   - No database query needed during execution

### Discovery Flow: Finding Child Executions

**Current limitation**: When stopping a parent execution, there's **no built-in way to discover child executions** because:

1. **No explicit parent reference** in database
2. **No execution tree tracking**
3. Only the AgentExecutorBlock instance knows its child graph_exec_id

**Workarounds used:**
- AgentExecutorBlock stores child graph_exec_id in local scope
- If parent fails, it calls `_stop()` on the child it spawned
- But if parent is forcefully terminated, the child may not be stopped

---

## 9. CASCADING STOP OPERATIONS

### Current Cascading Behavior

**Scenario 1: Parent Agent Stops (via API)**
```
User calls: POST /graphs/{parent_graph_id}/executions/{parent_exec_id}/stop
    ↓
stop_graph_execution(parent_exec_id)
    ↓
Publish CancelExecutionEvent to RabbitMQ
    ↓
Executor's cancel handler sets cancel_event
    ↓
Parent execution loop detects cancel.is_set()
    ↓
Breaks out of main loop
    ↓
Calls _cleanup_graph_execution()
    ↓
Cancels all running node tasks
    ↓
Updates parent status to TERMINATED
    ↓
Parent's AgentExecutorBlock._run() receives ExecutionEvent
    ↓
WHAT HAPPENS TO CHILD? 
    → Child continues executing (no cascading stop!)
```

**Scenario 2: Child Agent Stops**
```
If child is explicitly stopped: Works fine
    ↓
Child execution terminated
    ↓
Parent's event listener receives TERMINATED event
    ↓
Parent breaks from _run() loop
```

### Problem Statement

**Current Implementation:**
- When parent is stopped, it doesn't cascade to children
- Children continue running and consuming credits
- Event listener in AgentExecutorBlock is cancelled but child execution continues
- Only if AgentExecutorBlock catches an exception does it call `_stop()` on child

**Why This Happens:**
1. No database parent-child relationship
2. Parent termination doesn't trigger child termination
3. Child execution is independent RabbitMQ message
4. No way to discover all child executions from parent

---

## 10. EXECUTION STATE MANAGEMENT

### State Machine: Queued vs Running vs Stopped

```
INCOMPLETE
    ↓
QUEUED → waiting in RabbitMQ
    ↓ (executor picks up)
RUNNING → executing on worker
    ↓ (cancel event set)
TERMINATED ← execution cancelled
    ↓
clean up tasks
    ↓
Database updated

Alternative paths:
QUEUED → TERMINATED (via stop_graph_execution if not yet running)
RUNNING → TERMINATED (via cancel handler + cleanup)
RUNNING → FAILED (if error occurs)
RUNNING → COMPLETED (if succeeds)
```

### Execution Progress Tracking

**NodeExecutionProgress Class**: Tracks per-node async tasks

```python
class NodeExecutionProgress:
    def __init__(self):
        self.output: dict[str, list[ExecutionOutputEntry]] = defaultdict(list)
        self.tasks: dict[str, Future] = {}
        self._lock = threading.Lock()
    
    def stop(self) -> list[str]:
        """Stops all tasks and returns cancelled IDs"""
        cancelled_ids = []
        for task_id, task in self.tasks.items():
            if task.done():
                continue
            task.cancel()  # Cancel the async task
            cancelled_ids.append(task_id)
        return cancelled_ids
    
    def wait_for_done(self, timeout: float = 5.0):
        """Wait for all cancelled tasks to complete cancellation"""
        # Polls until all tasks are done or timeout
```

### RabbitMQ Queue Configuration

**File:** `/root/autogpt-1/autogpt_platform/backend/backend/executor/utils.py`
**Lines:** 551-608

```python
GRAPH_EXECUTION_EXCHANGE = Exchange(
    name="graph_execution",
    type=ExchangeType.DIRECT,
    durable=True,
    auto_delete=False,
)
GRAPH_EXECUTION_QUEUE_NAME = "graph_execution_queue"
GRAPH_EXECUTION_ROUTING_KEY = "graph_execution.run"

GRAPH_EXECUTION_CANCEL_EXCHANGE = Exchange(
    name="graph_execution_cancel",
    type=ExchangeType.FANOUT,  # Broadcasts to all workers
    durable=True,
    auto_delete=True,
)
GRAPH_EXECUTION_CANCEL_QUEUE_NAME = "graph_execution_cancel_queue"

GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 24 * 60 * 60  # 1 day to complete active executions

def create_execution_queue_config() -> RabbitMQConfig:
    run_queue = Queue(
        name=GRAPH_EXECUTION_QUEUE_NAME,
        exchange=GRAPH_EXECUTION_EXCHANGE,
        routing_key=GRAPH_EXECUTION_ROUTING_KEY,
        durable=True,
        auto_delete=False,
        arguments={
            # x-consumer-timeout disabled - let graphs run indefinitely
            "x-consumer-timeout": GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS * 1000,
        },
    )
    cancel_queue = Queue(
        name=GRAPH_EXECUTION_CANCEL_QUEUE_NAME,
        exchange=GRAPH_EXECUTION_CANCEL_EXCHANGE,
        routing_key="",  # not used for FANOUT
        durable=True,
        auto_delete=False,
    )
    return RabbitMQConfig(
        vhost="/",
        exchanges=[GRAPH_EXECUTION_EXCHANGE, GRAPH_EXECUTION_CANCEL_EXCHANGE],
        queues=[run_queue, cancel_queue],
    )
```

---

## 11. SUMMARY TABLE

| Component | Location | Purpose |
|-----------|----------|---------|
| REST Endpoint | `/server/routers/v1.py:922` | User-facing stop API |
| Stop Logic | `/executor/utils.py:615` | Core stop mechanism |
| AgentExecutor | `/blocks/agent.py:21` | Spawns child executions |
| Manager | `/executor/manager.py:781` | Runs execution loop |
| Cancel Handler | `/executor/manager.py:1415` | Receives cancel events |
| Cleanup | `/executor/manager.py:1055` | Terminates tasks |
| Schema | `schema.prisma:357` | Database execution models |

---

## 12. CURRENT LIMITATIONS & AREAS FOR IMPROVEMENT

### 1. No Explicit Parent-Child Relationship Tracking
- Cannot easily find all child executions of a parent
- Requires iterating all executions and filtering by timing

### 2. Cascading Stop Not Automatic
- Stopping parent doesn't stop children
- Child continues executing and consuming credits
- Only stopped if parent catches exception and calls `_stop()`

### 3. Event Bus Doesn't Track Parentage
- Event bus only knows graph_id, not parent graph_id
- Cannot efficiently filter by parent execution

### 4. Race Conditions
- Child execution may be created after parent stop signal
- Child gets queued but parent already terminated
- Child executes orphaned without supervision

### 5. Credit Usage
- Child executions charged independently
- Parent cannot cap total cost of sub-execution tree

---

## 13. RECOMMENDED IMPROVEMENTS

1. **Add Parent Execution Tracking**:
   ```prisma
   model AgentGraphExecution {
       ...
       parentGraphExecutionId String?
       ParentExecution AgentGraphExecution? @relation("ParentChild", fields: [parentGraphExecutionId], references: [id])
       ChildExecutions AgentGraphExecution[] @relation("ParentChild")
   }
   ```

2. **Implement Cascading Stop**:
   - When parent receives cancel event
   - Query database for all child executions
   - Publish stop messages for each child
   - Wait for all children to terminate

3. **Add Execution Context**:
   - Pass parent execution ID to child via AgentExecutorBlock.Input
   - Store in GraphExecutionEntry
   - Use for filtering and cascading

4. **Improve Event Bus**:
   - Track parent graph_exec_id in execution events
   - Allow filtering by parent execution
   - Publish parent-child relationship changes

5. **Timeout & Credit Protection**:
   - Set execution depth limits
   - Track total cost of execution tree
   - Enforce maximum nesting depth
