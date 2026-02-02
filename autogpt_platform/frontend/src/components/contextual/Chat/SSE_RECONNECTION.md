# SSE Reconnection Contract for Long-Running Operations

This document describes the client-side contract for handling SSE (Server-Sent Events) disconnections and reconnecting to long-running background tasks.

## Overview

When a user triggers a long-running operation (like agent generation), the backend:

1. Spawns a background task that survives SSE disconnections
2. Returns an `operation_started` response with a `task_id`
3. Stores stream messages in Redis Streams for replay

Clients can reconnect to the task stream at any time to receive missed messages.

## Client-Side Flow

### 1. Receiving Operation Started

When you receive an `operation_started` tool response:

```typescript
// The response includes a task_id for reconnection
{
  type: "operation_started",
  tool_name: "generate_agent",
  operation_id: "uuid-...",
  task_id: "task-uuid-...",  // <-- Store this for reconnection
  message: "Operation started. You can close this tab."
}
```

### 2. Storing Task Info

Use the chat store to track the active task:

```typescript
import { useChatStore } from "./chat-store";

// When operation_started is received:
useChatStore.getState().setActiveTask(sessionId, {
  taskId: response.task_id,
  operationId: response.operation_id,
  toolName: response.tool_name,
  lastMessageId: "0",
});
```

### 3. Reconnecting to a Task

To reconnect (e.g., after page refresh or tab reopen):

```typescript
const { reconnectToTask, getActiveTask } = useChatStore.getState();

// Check if there's an active task for this session
const activeTask = getActiveTask(sessionId);

if (activeTask) {
  // Reconnect to the task stream
  await reconnectToTask(
    sessionId,
    activeTask.taskId,
    activeTask.lastMessageId, // Resume from last position
    (chunk) => {
      // Handle incoming chunks
      console.log("Received chunk:", chunk);
    },
  );
}
```

### 4. Tracking Message Position

To enable precise replay, update the last message ID as chunks arrive:

```typescript
const { updateTaskLastMessageId } = useChatStore.getState();

function handleChunk(chunk: StreamChunk) {
  // If chunk has an index/id, track it
  if (chunk.idx !== undefined) {
    updateTaskLastMessageId(sessionId, String(chunk.idx));
  }
}
```

## API Endpoints

### Task Stream Reconnection

```
GET /api/chat/tasks/{taskId}/stream?last_message_id={idx}
```

- `taskId`: The task ID from `operation_started`
- `last_message_id`: Last received message index (default: "0" for full replay)

Returns: SSE stream of missed messages + live updates

## Chunk Types

The reconnected stream follows the same Vercel AI SDK protocol:

| Type                    | Description             |
| ----------------------- | ----------------------- |
| `start`                 | Message lifecycle start |
| `text-delta`            | Streaming text content  |
| `text-end`              | Text block completed    |
| `tool-output-available` | Tool result available   |
| `finish`                | Stream completed        |
| `error`                 | Error occurred          |

## Error Handling

If reconnection fails:

1. Check if task still exists (may have expired - default TTL: 1 hour)
2. Fall back to polling the session for final state
3. Show appropriate UI message to user

## Persistence Considerations

For robust reconnection across browser restarts:

```typescript
// Store in localStorage/sessionStorage
const ACTIVE_TASKS_KEY = "chat_active_tasks";

function persistActiveTask(sessionId: string, task: ActiveTaskInfo) {
  const tasks = JSON.parse(localStorage.getItem(ACTIVE_TASKS_KEY) || "{}");
  tasks[sessionId] = task;
  localStorage.setItem(ACTIVE_TASKS_KEY, JSON.stringify(tasks));
}

function loadPersistedTasks(): Record<string, ActiveTaskInfo> {
  return JSON.parse(localStorage.getItem(ACTIVE_TASKS_KEY) || "{}");
}
```

## Backend Configuration

The following backend settings affect reconnection behavior:

| Setting             | Default | Description                        |
| ------------------- | ------- | ---------------------------------- |
| `stream_ttl`        | 3600s   | How long streams are kept in Redis |
| `stream_max_length` | 1000    | Max messages per stream            |

## Testing

To test reconnection locally:

1. Start a long-running operation (e.g., agent generation)
2. Note the `task_id` from the `operation_started` response
3. Close the browser tab
4. Reopen and call `reconnectToTask` with the saved `task_id`
5. Verify that missed messages are replayed

See the main README for full local development setup.
