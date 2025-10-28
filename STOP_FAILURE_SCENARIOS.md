# Graph Execution Stop - Failure Scenarios Analysis

**Date:** October 28, 2025
**Branch:** feat/improve-stop-graph-execution

## Executive Summary

The stop broadcast mechanism uses RabbitMQ FANOUT exchange to broadcast `CancelExecutionEvent` to all executors. While the **parent execution itself stops correctly**, there are **critical failure scenarios where child executions continue running**. This analysis identifies all scenarios where the stop mechanism fails or behaves unexpectedly.

---

## Stop Mechanism Overview

### How Stop Works (Successful Path)

```
User calls: POST /graphs/{id}/executions/{exec_id}/stop
    ↓
utils.stop_graph_execution() publishes CancelExecutionEvent to RabbitMQ FANOUT
    ↓
ALL executors receive the broadcast (manager.py:1415 _handle_cancel_message)
    ↓
If executor has this graph_exec_id in active_graph_runs:
    ↓
    Sets cancel_event threading.Event
    ↓
Main execution loop checks cancel.is_set() at 3 points (manager.py:850, 927, 935)
    ↓
Loop breaks, cleanup runs, status → TERMINATED
```

### Key Components

1. **Broadcast**: RabbitMQ FANOUT exchange (all workers receive message)
2. **Detection**: `_handle_cancel_message()` checks if execution is active locally
3. **Propagation**: Sets `threading.Event` flag
4. **Cleanup**: Stops node executions, evaluations, updates DB

---

## CRITICAL FAILURE SCENARIOS

### Scenario 1: Child Executions Are Not Stopped ⚠️ **CRITICAL**

**What Happens:**
- Parent execution has AgentExecutorBlock that spawned child execution
- User stops parent execution
- Parent's cancel event is set, parent stops
- **Child execution continues running indefinitely**

**Why This Fails:**

1. **No parent-child relationship in database**
   - Child execution has no `parentGraphExecutionId` field
   - No way to query "all children of execution X"

2. **No cascading stop logic**
   - `stop_graph_execution()` only publishes ONE cancel event for ONE execution ID
   - Doesn't discover or stop children

3. **AgentExecutorBlock only stops child on exception**
   - Code location: `blocks/agent.py:99-108`
   - Only stops child if `BaseException` caught during `_run()`
   - Clean parent termination doesn't trigger exception → child keeps running

**Code Evidence:**

```python
# blocks/agent.py:68-108
async def run(self, input_data: Input, **kwargs) -> BlockOutput:
    graph_exec = await execution_utils.add_graph_execution(...)  # Child created

    try:
        async for name, data in self._run(...):  # Listening to child events
            yield name, data
    except BaseException as e:  # ← ONLY stops child on exception
        await self._stop(graph_exec_id=graph_exec.id, ...)
        raise
    # If parent is cancelled cleanly, no exception → child NOT stopped
```

**Impact:**
- **Orphaned child executions** consuming credits
- **No user control** over child execution tree
- **Cost overruns** without limit
- **Resource leaks** from abandoned executions

**Reproduction:**
```
1. Create parent graph with AgentExecutorBlock
2. AgentExecutorBlock spawns child graph execution
3. While child is RUNNING or QUEUED, stop parent
4. Parent stops immediately
5. Child continues executing ← BUG
```

---

### Scenario 2: Race Condition - Child Created After Parent Stop

**What Happens:**
- Parent execution is in the middle of stopping
- AgentExecutorBlock.run() calls `add_graph_execution()` to spawn child
- Parent's cancel event is already set
- Child gets created and queued in RabbitMQ
- Parent completes termination
- **Child starts executing after parent is already terminated**

**Why This Fails:**

1. **No pre-flight cancellation check** in AgentExecutorBlock
   - Doesn't check if parent is being cancelled before spawning child
   - Code location: `blocks/agent.py:72-78`

2. **Async timing issue**
   - Parent's `cancel.is_set()` happens in executor thread
   - AgentExecutorBlock runs in async node evaluation loop
   - No shared cancellation token between them

**Code Evidence:**

```python
# blocks/agent.py:68-78
async def run(self, input_data: Input, **kwargs) -> BlockOutput:
    # NO check if parent is being cancelled
    graph_exec = await execution_utils.add_graph_execution(...)  # Child created regardless

    try:
        async for name, data in self._run(...):
            yield name, data
```

**Timeline:**
```
T0: Parent execution running
T1: User calls stop on parent
T2: CancelEvent published to RabbitMQ
T3: Executor receives cancel, sets cancel_event flag
T4: AgentExecutorBlock.run() called (before cancel.is_set() check in main loop)
T5: Child execution created and queued  ← Race condition
T6: Parent loop checks cancel.is_set(), breaks
T7: Parent cleaned up and terminated
T8: Child execution starts on executor  ← Orphaned
```

**Impact:**
- Child executions start after parent already stopped
- Invisible to user (parent shows TERMINATED)
- Consumes credits unexpectedly

---

### Scenario 3: Child in QUEUED Status When Parent Stops

**What Happens:**
- AgentExecutorBlock spawns child execution
- Child is in QUEUED status (waiting in RabbitMQ for executor)
- Parent gets stopped
- Parent terminates immediately
- **Child eventually gets picked up by executor and starts running**

**Why This Fails:**

1. **QUEUED executions are independent messages**
   - Already in RabbitMQ queue, waiting for worker
   - Stop broadcast only affects ACTIVE executions
   - Code location: `manager.py:1433-1437`

2. **No parent context in queue message**
   - RabbitMQ message is `GraphExecutionEntry`
   - No `parentGraphExecutionId` field to check
   - Executor can't know "this child's parent was cancelled"

**Code Evidence:**

```python
# manager.py:1433-1437
if graph_exec_id not in self.active_graph_runs:
    logger.debug(f"Cancel received for {graph_exec_id} but not active.")
    return  # ← Queued child not in active_graph_runs, cancel ignored
```

**Impact:**
- Delayed orphan executions
- Appears to work initially (parent stops) but child starts later
- Hard to debug (why is my stopped graph still running?)

---

### Scenario 4: Event Bus Listener Not Cancelled Properly

**What Happens:**
- AgentExecutorBlock._run() listens to child events via `event_bus.listen()`
- Parent execution is cancelled
- Parent's async generator loop breaks
- **Event bus listener may not unsubscribe immediately**

**Why This Might Fail:**

1. **Async generator cleanup timing**
   - Code location: `blocks/agent.py:128-152`
   - Uses `async for event in event_bus.listen()`
   - Generator cleanup happens on exception or break
   - Redis pubsub unsubscribe may be delayed

2. **No explicit cancellation of listener**
   - Relies on async context manager behavior
   - If parent terminates abruptly, listener may stay subscribed

**Code Evidence:**

```python
# blocks/agent.py:128-152
async for event in event_bus.listen(
    user_id=user_id,
    graph_id=graph_id,
    graph_exec_id=graph_exec_id,
):
    if event.status in [COMPLETED, TERMINATED, FAILED]:
        break  # ← Normal exit, cleanup happens
    # But if parent is cancelled, this async generator is just abandoned
```

**Impact:**
- Memory leak from unclosed Redis connections
- Orphaned pubsub subscriptions
- Potential message delivery to terminated context

---

### Scenario 5: Multiple Executor Workers - Wrong Worker Receives Cancel

**What Happens:**
- Execution running on Executor Worker A
- User stops execution
- Cancel broadcast sent to ALL workers via FANOUT
- **Worker B, C, D receive cancel but execution not active there**
- Worker A receives cancel ✓ but delayed

**Why This Might Fail:**

1. **RabbitMQ message ordering not guaranteed**
   - FANOUT broadcasts to all, but arrival order varies
   - Network latency between workers

2. **Race between cancel and execution migration**
   - If execution crashes on Worker A and restarts on Worker B
   - Cancel might arrive at old worker

**Code Evidence:**

```python
# manager.py:1433-1437
if graph_exec_id not in self.active_graph_runs:
    logger.debug(f"Cancel received for {graph_exec_id} but not active.")
    return  # ← Most workers ignore cancel (correct behavior)
```

**Impact:**
- Usually works correctly (broadcast reaches correct worker)
- Potential delay if network issues
- No retry mechanism if cancel message lost

---

### Scenario 6: Execution Already in Cleanup Phase

**What Happens:**
- Execution is completing normally (almost done)
- User stops execution at same moment
- Execution transitions to COMPLETED
- **Cancel arrives but execution already finished**

**Why This Is Okay (Not Really a Failure):**

1. **Terminal status check handles this**
   - Code location: `utils.py:647-654`
   - Checks if already TERMINATED, COMPLETED, or FAILED
   - Returns immediately if so

**Code Evidence:**

```python
# utils.py:647-654
if graph_exec.status in [
    ExecutionStatus.TERMINATED,
    ExecutionStatus.COMPLETED,
    ExecutionStatus.FAILED,
]:
    await get_async_execution_event_bus().publish(graph_exec)
    return  # ← Safe exit
```

**Impact:**
- No negative impact
- Handled correctly by existing code

---

### Scenario 7: Timeout on Stop (15 Second Default)

**What Happens:**
- User stops execution
- Execution is RUNNING with long-running operations
- 15 second timeout expires
- **TimeoutError raised to user, but execution might still be stopping**

**Why This Happens:**

1. **Hardcoded wait_timeout**
   - Code location: `utils.py:618`
   - Default 15 seconds
   - Some operations take longer to clean up

2. **Cleanup continues in background**
   - Timeout is just for polling status
   - Actual cleanup keeps running

**Code Evidence:**

```python
# utils.py:678-681
raise TimeoutError(
    f"Graph execution #{graph_exec_id} will need to take longer than {wait_timeout} seconds to stop."
)
# But cleanup keeps running in executor thread
```

**Impact:**
- User sees error but execution eventually stops
- Confusing UX ("did it stop or not?")
- Status polling required to confirm

---

### Scenario 8: Deep Nesting - Multi-Level Child Executions

**What Happens:**
- Parent execution spawns Child A
- Child A spawns Child B
- Child B spawns Child C
- User stops parent
- **Only parent stops, all children continue**

**Why This Fails:**

1. **No recursive stop logic**
   - Same as Scenario 1, but worse
   - No way to discover grandchildren, great-grandchildren, etc.

2. **No execution tree limit**
   - Users can create arbitrarily deep nesting
   - Each level compounds the orphaning problem

**Impact:**
- Exponentially worse than Scenario 1
- Entire execution trees abandoned
- Massive credit consumption

---

## Edge Cases Summary Table

| Scenario | Severity | Current Behavior | Expected Behavior |
|----------|----------|------------------|-------------------|
| 1. Child executions not stopped | **CRITICAL** | Children keep running | Children should stop with parent |
| 2. Race - child created after stop | **HIGH** | Child starts orphaned | Should detect parent cancelled |
| 3. Child in QUEUED when parent stops | **HIGH** | Child starts later | Should be removed from queue |
| 4. Event bus listener leak | **MEDIUM** | May leave subscriptions open | Should cleanup immediately |
| 5. Wrong worker receives cancel | **LOW** | Works correctly (broadcast) | (Already working) |
| 6. Already completed execution | **N/A** | Handled correctly | (Already working) |
| 7. Stop timeout | **MEDIUM** | User sees error | Better UX needed |
| 8. Deep nesting | **CRITICAL** | All children orphaned | Recursive stop |

---

## Additional Failure Modes

### Database Transaction Failures

**Scenario:** Stop called, cancel event published, but DB update fails

**Location:** `utils.py:664-672`

```python
await asyncio.gather(
    db.update_graph_execution_stats(
        graph_exec_id=graph_exec.id,
        status=ExecutionStatus.TERMINATED,
    ),
    get_async_execution_event_bus().publish(graph_exec),
)
```

**Impact:**
- Execution stopped but status not updated
- Shows RUNNING but actually terminated
- Billing might be affected

---

### RabbitMQ Connection Loss

**Scenario:** Network partition, RabbitMQ down, cancel broadcast fails

**Impact:**
- Cancel event never reaches executor
- Execution continues running
- No retry mechanism

---

### Redis Connection Loss (Event Bus)

**Scenario:** Redis down, event bus publish fails

**Location:** Called after successful stop `utils.py:653`

**Impact:**
- Frontend doesn't receive termination event
- UI shows stale status
- User thinks execution still running

---

## Recommended Fixes

### Fix #1: Add Parent-Child Tracking (Database Schema)

**Changes Needed:**

1. Add to `schema.prisma:357`:
```prisma
model AgentGraphExecution {
  ...
  parentGraphExecutionId String?
  ParentExecution AgentGraphExecution? @relation("ParentChild", fields: [parentGraphExecutionId], references: [id])
  ChildExecutions AgentGraphExecution[] @relation("ParentChild")
}
```

2. Pass parent ID in `add_graph_execution()` calls from AgentExecutorBlock

---

### Fix #2: Implement Cascading Stop

**Changes Needed in `utils.py:615`:**

```python
async def stop_graph_execution(
    user_id: str,
    graph_exec_id: str,
    wait_timeout: float = 15.0,
    cascade: bool = True,  # New parameter
):
    # First, stop the specified execution
    await _stop_single_execution(user_id, graph_exec_id, wait_timeout)

    if cascade:
        # Query all child executions
        children = await db.get_child_executions(graph_exec_id)

        # Recursively stop all children
        await asyncio.gather(*[
            stop_graph_execution(user_id, child.id, wait_timeout, cascade=True)
            for child in children
        ])
```

---

### Fix #3: Pre-flight Cancellation Check in AgentExecutorBlock

**Changes Needed in `blocks/agent.py:68`:**

```python
async def run(self, input_data: Input, **kwargs) -> BlockOutput:
    # Check if parent execution is being cancelled
    if hasattr(kwargs, 'cancel_event') and kwargs['cancel_event'].is_set():
        raise CancelledError("Parent execution cancelled before child spawn")

    graph_exec = await execution_utils.add_graph_execution(
        parent_graph_exec_id=kwargs.get('graph_exec_id'),  # Pass parent ID
        ...
    )
```

---

### Fix #4: Prevent QUEUED Children from Starting

**Changes Needed in `manager.py:1448` (run message handler):**

```python
def _handle_run_message(self, ...):
    request = GraphExecutionEntry.model_validate_json(body)

    # NEW: Check if parent is already terminated
    if request.parent_graph_exec_id:
        parent = db_client.get_graph_execution_meta(
            execution_id=request.parent_graph_exec_id,
            user_id=request.user_id,
        )
        if parent and parent.status == ExecutionStatus.TERMINATED:
            logger.info(f"Skipping execution {request.graph_exec_id} - parent terminated")
            db_client.update_graph_execution_stats(
                graph_exec_id=request.graph_exec_id,
                status=ExecutionStatus.TERMINATED,
            )
            return  # Don't execute

    # Continue with normal execution...
```

---

## Testing Requirements

### Test Cases to Add

1. **test_stop_with_child_execution**
   - Parent spawns child
   - Stop parent while child RUNNING
   - Assert child also stops

2. **test_stop_with_queued_child**
   - Parent spawns child
   - Stop parent while child QUEUED
   - Assert child never starts

3. **test_race_condition_child_spawn**
   - Stop parent during AgentExecutorBlock.run()
   - Assert child not created or immediately stopped

4. **test_deep_nesting_stop**
   - Parent → Child → Grandchild (3 levels)
   - Stop parent
   - Assert all descendants stop

5. **test_multiple_children_stop**
   - Parent spawns 5 children
   - Stop parent
   - Assert all 5 children stop

---

## Performance Considerations

### Current Performance
- Single execution stop: O(1) - just one cancel event
- Wait timeout: Up to 15 seconds

### With Cascading Stop
- Tree traversal: O(n) where n = total nodes in execution tree
- Parallel stop: O(depth) with asyncio.gather
- Worst case: Deep tree with many nodes = slower stop

**Mitigation:**
- Limit maximum tree depth (e.g., 10 levels)
- Limit maximum children per execution (e.g., 100)
- Add configurable timeout per level

---

## Conclusion

The current stop mechanism **works correctly for single executions** but has **critical gaps for nested executions via AgentExecutorBlock**. The broadcast mechanism itself is sound, but the lack of parent-child relationship tracking means:

1. ❌ Child executions are never stopped
2. ❌ QUEUED children start after parent termination
3. ❌ No way to stop execution trees
4. ❌ Uncontrolled credit consumption

**Priority Fixes:**
1. **CRITICAL**: Add parent-child tracking (Schema change)
2. **CRITICAL**: Implement cascading stop logic
3. **HIGH**: Prevent queued children from starting
4. **MEDIUM**: Add pre-flight cancellation check

All fixes are achievable without breaking changes to existing API.
