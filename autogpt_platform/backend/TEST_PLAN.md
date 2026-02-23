# Copilot Streaming Issues - Test Plan

## Issues to Fix & Test

### Issue #1: Stream timeout toast appearing on every chat
**Expected**: No timeout toast
**Test**: Send "hello", verify no toast appears within 12 seconds
**Root Cause**: subscribe_to_task hanging or returning None

### Issue #2: Chat not loading response unless retried/refreshed
**Expected**: Response appears immediately without refresh
**Test**: Send message, verify response appears without page refresh
**Root Cause**: SSE stream not delivering chunks to frontend

### Issue #3: Updates happening in batch instead of real-time streaming
**Expected**: Text streams token-by-token as it's generated
**Test**: Send message, watch response appear gradually (not all at once)
**Root Cause**: Chunks being buffered instead of streamed

### Issue #4: Session stuck on red button (loading state)
**Expected**: Loading indicator disappears when response completes
**Test**: Send message, verify loading state clears after response
**Root Cause**: StreamFinish not being received or processed

### Issue #5: Agent execution dropping after first tool call
**Expected**: Agent continues through multiple tool calls
**Test**: Ask agent to create something (triggers tool), verify completion
**Root Cause**: Tool execution not resuming stream properly

### Issue #6: Second chat showing introduction again
**Expected**: Follow-up messages maintain context
**Test**: Say "hello", then "what's my name", verify no repeated intro
**Root Cause**: Session history not being maintained or passed correctly

## Test Sequence

1. Start backend with all fixes
2. Start frontend
3. Open browser to http://localhost:3000
4. Test each issue in order
5. Document results
6. Fix any failures
7. Re-test until all pass
