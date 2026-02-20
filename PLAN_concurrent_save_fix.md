# Plan: Defense-in-Depth Fix for Concurrent Message Saves

## Problem Summary

**Current Issue:** Race condition between two concurrent writers during SDK streaming causes unique constraint violations on `(sessionId, sequence)`:

1. **Streaming loop** (sdk/service.py:720-930): Tracks `saved_msg_count` in-memory
2. **Long-running callback** (sdk/service.py:210): Calls `upsert_chat_session(session)` WITHOUT `existing_message_count`

**Race Sequence:**
```
1. Streaming: saved_msg_count = 5, DB has 5 messages
2. Agent calls create_agent → callback appends msg → saves at sequence 5 → DB has 6
3. Streaming: saved_msg_count still 5 (STALE!)
4. Incremental save tries to insert at sequence 5 → 💥 UNIQUE CONSTRAINT
```

## Solution: Three Complementary Layers

### ✅ Layer 1: Upsert (DONE - Already in PR)
**File:** `backend/copilot/db.py`
**Status:** ✅ Implemented and merged

Changes:
- `add_chat_messages_batch` uses `upsert()` instead of `create()`
- Explicitly constructs `update_data` with only updateable fields
- Excludes `Session` relation and `sequence` (part of unique key)

**Benefit:** Final safety net - if duplicate sequence reached, update instead of crash

### 🔧 Layer 2: Query Latest Sequence Before Each Save (NEW)
**Files:** `backend/copilot/db.py`, `backend/copilot/sdk/service.py`

**Changes to db.py:**
Add helper function to get current message count:
```python
async def get_chat_session_message_count(session_id: str) -> int:
    """Get the current count of messages in a session from DB (source of truth)."""
    count = await PrismaChatMessage.prisma().count(
        where={"sessionId": session_id}
    )
    return count
```

**Changes to sdk/service.py:**
Before each `upsert_chat_session`, query DB for latest count:

1. **Line 720** - Initialize from DB instead of `len(session.messages)`:
   ```python
   # Track persisted message count. Query DB as source of truth.
   saved_msg_count = await chat_db().get_chat_session_message_count(session_id)
   ```

2. **Line 898-902** - Query before incremental save:
   ```python
   try:
       # Query DB for latest count before save (defense against stale counter)
       db_count = await chat_db().get_chat_session_message_count(session_id)
       await upsert_chat_session(
           session,
           existing_message_count=db_count,
       )
       saved_msg_count = len(session.messages)
   except Exception as save_err:
       ...
   ```

3. **Line 926-930** - Query before incremental save:
   ```python
   try:
       # Query DB for latest count before save (defense against stale counter)
       db_count = await chat_db().get_chat_session_message_count(session_id)
       await upsert_chat_session(
           session,
           existing_message_count=db_count,
       )
       saved_msg_count = len(session.messages)
   except Exception as save_err:
       ...
   ```

**Benefit:** DB is source of truth - prevents using stale in-memory counter

**Trade-off:** Extra DB query (simple COUNT, very fast ~1-2ms)

### 🔧 Layer 3: Share Counter with Long-Running Callback (NEW)
**File:** `backend/copilot/sdk/service.py`

**Change the callback to receive and update shared counter:**

1. **Line 136** - Modify callback signature to accept shared counter:
   ```python
   def _build_long_running_callback(
       user_id: str | None,
       saved_msg_count_ref: list[int],  # NEW: mutable reference
   ) -> LongRunningCallback:
   ```

2. **Line 210** - Use shared counter and update it:
   ```python
   session.messages.append(pending_message)
   # Query DB for latest count (Layer 2)
   db_count = await chat_db().get_chat_session_message_count(session_id)
   await upsert_chat_session(session, existing_message_count=db_count)
   # Update shared counter (Layer 3)
   saved_msg_count_ref[0] = len(session.messages)
   ```

3. **Line 720** - Pass mutable reference to callback:
   ```python
   # Make counter a mutable list so callback can update it
   saved_msg_count_ref = [len(session.messages)]
   saved_msg_count = saved_msg_count_ref[0]

   # Build callback with shared counter
   long_running_callback = _build_long_running_callback(
       user_id=user_id,
       saved_msg_count_ref=saved_msg_count_ref,
   )
   ```

4. **Lines 898-902, 926-930** - Read from shared ref:
   ```python
   # Query DB for latest count (Layer 2)
   db_count = await chat_db().get_chat_session_message_count(session_id)
   await upsert_chat_session(session, existing_message_count=db_count)
   # Update shared counter (Layer 3)
   saved_msg_count_ref[0] = len(session.messages)
   saved_msg_count = saved_msg_count_ref[0]
   ```

**Benefit:** Both writers coordinate via shared counter - in-memory tracking stays accurate

## Why All Three Layers?

**Defense in Depth:**
- **Layer 2 alone** solves the race but adds DB queries (performance cost)
- **Layer 3 alone** solves coordination but assumes no other writers exist
- **Layer 1 alone** prevents crashes but may silently overwrite data

**Together:**
- Layer 2 (query DB) = Correctness guarantee (DB is truth)
- Layer 3 (shared counter) = Performance optimization (reduces DB queries in common case)
- Layer 1 (upsert) = Final safety net (prevents crashes from any remaining edge cases)

## Implementation Order

1. ✅ **Layer 1 (upsert)** - DONE, already in PR #12177
2. **Layer 2 (query DB)** - Implement next (adds `get_chat_session_message_count`)
3. **Layer 3 (shared counter)** - Implement last (modifies callback signature)

## Testing

After implementation:
1. **Unit test**: Concurrent saves with simulated race
2. **Manual test**: Trigger create_agent during streaming, verify no errors
3. **Sentry**: Monitor for unique constraint errors (should drop to zero)

## Files Modified

- ✅ `backend/copilot/db.py` - Layer 1 done, Layer 2 add helper
- `backend/copilot/sdk/service.py` - Layers 2 & 3

## Verification

After all layers:
- No more `Unique constraint failed on (sessionId, sequence)` errors
- Incremental saves work correctly even with concurrent writers
- No message loss or data corruption
- Performance acceptable (COUNT query is <2ms)
