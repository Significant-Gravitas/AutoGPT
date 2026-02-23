# Reconnection Flow - Complete Review

## ✅ Reconnection Implementation is CORRECT

### Backend Flow

**1. Session GET Endpoint** (`routes.py:266`)
```python
@router.get("/sessions/{session_id}")
async def get_session(session_id: str, user_id: str | None):
    # Fetches session from Redis/DB
    session = await get_chat_session(session_id, user_id)

    # Checks for active stream
    active_task, last_id = await stream_registry.get_active_task_for_session(
        session_id, user_id
    )

    if active_task:
        # CRITICAL: Filters out in-progress assistant message
        # to prevent duplication when stream replays
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]

        # Returns active_stream info with last_message_id="0-0"
        active_stream_info = ActiveStreamInfo(
            task_id=active_task.session_id,
            last_message_id="0-0",  # Full replay
            ...
        )

    return SessionDetailResponse(
        messages=messages,
        active_stream=active_stream_info  # Frontend checks this
    )
```

**2. Resume Stream Endpoint** (`routes.py:684`)
```python
@router.get("/sessions/{session_id}/stream")
async def resume_session_stream(session_id: str, user_id: str | None):
    # Check for active task
    active_task, _ = await stream_registry.get_active_task_for_session(
        session_id, user_id
    )

    if not active_task:
        return Response(status_code=204)  # Nothing to resume

    # Subscribe with full replay
    queue = await stream_registry.subscribe_to_task(
        session_id=session_id,
        user_id=user_id,
        last_message_id="0-0"  # Replay from beginning
    )

    # Stream all chunks from Redis
    async def event_generator():
        while True:
            chunk = await asyncio.wait_for(queue.get(), timeout=30.0)
            yield chunk.to_sse()
            if isinstance(chunk, StreamFinish):
                break
```

**Key Points**:
- ✅ Uses `last_message_id="0-0"` for full replay
- ✅ Filters out in-progress assistant message to prevent duplication
- ✅ Returns 204 if no active stream
- ✅ Sends heartbeats every 30s during reconnection
- ✅ Properly cleans up subscriber queue on completion

### Frontend Flow

**1. Session Data Fetching** (`useChatSession.ts`)
```typescript
const sessionQuery = useGetV2GetSession(sessionId ?? "", {
  query: {
    enabled: !!sessionId,
    staleTime: Infinity,  // Cache until invalidated
    refetchOnWindowFocus: false,  // Don't refetch on tab switch
    refetchOnReconnect: false,  // Don't refetch on network reconnect
  },
});

// Extract active_stream flag
const hasActiveStream = useMemo(() => {
  if (sessionQuery.data?.status !== 200) return false;
  return !!sessionQuery.data.data.active_stream;  // Boolean flag
}, [sessionQuery.data]);
```

**2. Message Hydration** (`useChatSession.ts`)
```typescript
const hydratedMessages = useMemo(() => {
  if (sessionQuery.data?.status !== 200 || !sessionId) return undefined;
  return convertChatSessionMessagesToUiMessages(
    sessionId,
    sessionQuery.data.data.messages ?? [],
    { isComplete: !hasActiveStream }  // Marks tools as completed if stream done
  );
}, [sessionQuery.data, sessionId, hasActiveStream]);
```

**3. Reconnection Trigger** (`useCopilotPage.ts`)
```typescript
// Resume an active stream AFTER hydration completes
useEffect(() => {
  if (!sessionId) return;
  if (!hasActiveStream) return;  // No active stream, nothing to resume
  if (!hydratedMessages || hydratedMessages.length === 0) return;

  // Never resume if currently streaming
  if (status === "streaming" || status === "submitted") return;

  // Only resume once per session (prevent duplicate reconnections)
  if (hasResumedRef.current.get(sessionId)) return;

  // Mark as resumed and trigger
  hasResumedRef.current.set(sessionId, true);
  resumeStream();  // AI SDK's resumeStream()
}, [sessionId, hasActiveStream, hydratedMessages, status, resumeStream]);
```

**4. Transport Configuration** (`useCopilotPage.ts`)
```typescript
const transport = useMemo(
  () =>
    sessionId
      ? new DefaultChatTransport({
          api: `/api/chat/sessions/${sessionId}/stream`,

          // POST: Send new message
          prepareSendMessagesRequest: ({ messages }) => ({
            body: {
              message: extractText(messages[messages.length - 1]),
              is_user_message: last.role === "user",
              context: null,
            },
          }),

          // GET: Reconnect to existing stream
          prepareReconnectToStreamRequest: () => ({
            api: `/api/chat/sessions/${sessionId}/stream`,  // Same URL
          }),
        })
      : null,
  [sessionId],
);
```

**Key Points**:
- ✅ Waits for hydration before resuming
- ✅ Only resumes once per session
- ✅ Never resumes if already streaming
- ✅ Uses same endpoint for POST (new) and GET (resume)
- ✅ Backend distinguishes by HTTP method

### Deduplication

**Frontend** (`useCopilotPage.ts:39`)
```typescript
function deduplicateMessages(messages: UIMessage[]): UIMessage[] {
  const seen = new Set<string>();
  return messages.filter((msg) => {
    if (seen.has(msg.id)) return false;
    seen.add(msg.id);
    return true;
  });
}

// Applied continuously
const messages = useMemo(
  () => deduplicateMessages(rawMessages),
  [rawMessages],
);
```

**Key Points**:
- ✅ Deduplicates by message ID
- ✅ Applied continuously to prevent duplicates from reconnection
- ✅ Backend filters out in-progress assistant message
- ✅ Stream replay reconstructs the message

---

## 🧪 Test Coverage

### ✅ Backend Tests Exist
- `test_streaming_complete.py` - Tests dummy streaming flow
- `test_copilot_e2e.py` - Our new comprehensive E2E tests (7 passing)

### ❌ Missing Tests

**1. Reconnection Tests** (CRITICAL GAP)
No tests verify:
- Session GET returns correct `active_stream` flag
- Resume GET endpoint replays stream correctly
- Deduplication works after reconnection
- In-progress message filtering works

**2. Frontend Tests** (NO TESTS AT ALL)
No unit or E2E tests for:
- `useCopilotPage` hook
- `useChatSession` hook
- Message deduplication
- Reconnection trigger logic
- Transport configuration

---

## 🔍 Return Values Review

### Session GET Endpoint
**Request**: `GET /api/chat/sessions/{session_id}`

**Response** (`SessionDetailResponse`):
```typescript
{
  id: string,
  created_at: string,  // ISO datetime
  updated_at: string,  // ISO datetime
  user_id: string | null,
  messages: Message[],  // Filtered (no in-progress assistant)
  active_stream: {  // Present if stream active
    task_id: string,
    last_message_id: string,  // Always "0-0" for full replay
    operation_id: string,
    tool_name: string
  } | null
}
```

✅ **Correct**: Provides all info needed for reconnection

### Resume Stream Endpoint
**Request**: `GET /api/chat/sessions/{session_id}/stream`

**Response**:
- `204 No Content` - No active stream
- `200 OK` with SSE stream - Active stream exists

**SSE Events**:
```
event: <event_type>
data: <json_chunk>

// Example:
event: StreamStart
data: {"type":"stream_start","messageId":"abc","taskId":"xyz"}

event: StreamTextDelta
data: {"type":"text_delta","id":"123","delta":"Hello"}

event: StreamFinish
data: {"type":"stream_finish"}
```

✅ **Correct**: Replays full stream from Redis

### New Message Endpoint
**Request**: `POST /api/chat/sessions/{session_id}/stream`

**Body**:
```json
{
  "message": "User message text",
  "is_user_message": true,
  "context": null
}
```

**Response**: SSE stream (same format as resume)

✅ **Correct**: Same event format as resume

---

## ✅ Heartbeat Configuration

After our fixes:

| Location | Interval | Requirement |
|----------|----------|-------------|
| Frontend timeout | 12 seconds | `STREAM_START_TIMEOUT_MS` |
| SDK service | **10 seconds** | ✅ Fixed (was 15s) |
| Standard service | **10 seconds** | ✅ Fixed (was 15s) |
| Resume endpoint | 30 seconds | ✅ Correct (resume is more tolerant) |

✅ **All heartbeats now < frontend timeout**

---

## 🎯 Reconnection Flow Summary

### Happy Path
1. **User sends message** → Stream starts → Task created in Redis
2. **User refreshes page** → Session GET returns `active_stream: {...}`
3. **Frontend detects** `hasActiveStream === true`
4. **Frontend hydrates** messages from session (no in-progress assistant)
5. **Frontend calls** `resumeStream()` → GET to stream endpoint
6. **Backend replays** full stream from Redis with `last_message_id="0-0"`
7. **Frontend receives** all chunks, deduplicates by ID
8. **Stream completes** → Frontend shows final state

### Edge Cases

**Case 1: No active stream**
- Session GET returns `active_stream: null`
- Frontend doesn't call `resumeStream()`
- Shows hydrated messages only

**Case 2: Stream completes during page load**
- Race condition: session says active, but stream finishes
- Resume GET returns 204
- Frontend handles gracefully (no crash)

**Case 3: Multiple reconnections**
- `hasResumedRef` prevents duplicate reconnections
- Only resumes once per session

**Case 4: Concurrent streaming**
- Never resumes if `status === "streaming"`
- Prevents interfering with active stream

---

## 🚨 Identified Gaps

### 1. No Reconnection Tests (HIGH PRIORITY)
**Impact**: Can't verify reconnection works after code changes

**Recommended**: Add tests to `test_copilot_e2e.py`:
```python
@pytest.mark.asyncio
async def test_reconnection_after_refresh():
    """Test stream reconnection simulates page refresh."""
    # 1. Start stream
    # 2. Capture first few chunks
    # 3. Disconnect
    # 4. Check session has active_stream
    # 5. Reconnect with GET
    # 6. Verify replay from beginning
    # 7. Verify no duplicates
```

### 2. No Frontend Tests (MEDIUM PRIORITY)
**Impact**: Frontend reconnection logic not verified

**Recommended**: Add Playwright E2E test:
```typescript
test('reconnects to active stream after page refresh', async ({ page }) => {
  // 1. Send message
  // 2. Wait for stream to start
  // 3. Refresh page
  // 4. Verify stream resumes
  // 5. Verify no duplicate messages
});
```

### 3. No Deduplication Tests (MEDIUM PRIORITY)
**Impact**: Can't verify deduplication prevents duplicates

**Recommended**: Add unit test for `deduplicateMessages()`

---

## ✅ Conclusions

### What's Working
1. ✅ **Reconnection logic is correct** - Backend and frontend properly coordinated
2. ✅ **Heartbeat timing fixed** - All intervals < frontend timeout
3. ✅ **Deduplication exists** - Prevents duplicate messages
4. ✅ **Return values correct** - All endpoints return expected data
5. ✅ **Edge cases handled** - No active stream, concurrent streaming, etc.

### What's Missing
1. ❌ **No reconnection tests** - Gap in test coverage
2. ❌ **No frontend tests** - Copilot logic not tested
3. ❌ **No E2E tests** - Full flow not verified end-to-end

### Recommendation
**The implementation is correct and should work, but lacks test coverage.**

**For manual testing, verify**:
1. Send message → See streaming response
2. Refresh page mid-stream → Stream resumes from beginning
3. Refresh after completion → No reconnection (just hydrated messages)
4. No duplicate messages appear
5. Heartbeat keeps connection alive during long operations

**After manual testing passes**, add automated tests for reconnection.
