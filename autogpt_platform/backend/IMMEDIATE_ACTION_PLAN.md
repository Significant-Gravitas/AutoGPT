# Immediate Action Plan - Get Copilot Working

## Phase 1: Quick Wins (TODAY) ⚡

### 1. Fix Heartbeat Timing (5 min)
**Problem:** Frontend times out after 12s, backend sends heartbeats every 15s
**Fix:**
```python
# stream_registry.py line 505
messages = await redis.xread(
    {stream_key: current_id}, block=5000, count=100  # Change to 10000 (10s)
)
```

### 2. Manual Test All Issues (30 min)
**Test checklist:**
- [ ] Issue #34: Chat loads response first try
- [ ] Issue #35: Updates stream in real-time (not batched)
- [ ] Issue #37: Agent execution completes all tools
- [ ] Issue #38: New chat doesn't repeat intro
- [ ] Issue #40: Context maintained between messages
- [ ] Issue #41: No SDK warnings in console

**How to test:**
1. Start backend: `poetry run app`
2. Start frontend: `pnpm dev`
3. Test each scenario
4. Collect logs for failures

### 3. Check Message History in LLM Calls (15 min)
**Problem:** Context not maintained (Issue #40)
**Check:**
```python
# service.py - when calling LLM
# Verify messages array includes full history
logger.info(f"Calling LLM with {len(messages)} messages")
```

---

## Phase 2: Simplify Architecture (THIS WEEK) 🧹

### Day 1: Map Current Flow

**Create execution flow diagram:**
1. Trace one request from HTTP → response
2. Document every layer/file touched
3. Identify unnecessary hops

**Files to trace:**
- `routes.py` → where does it go?
- `service.py` → what does it call?
- `executor/processor.py` → why?
- `sdk/service.py` → why?
- `tool_adapter.py` → why?

### Day 2: Remove Dead Code

**Candidates for removal:**
1. ~~Long-running tool async infrastructure~~ ✅ DONE
2. `executor/processor.py` - if not needed
3. `sdk/tool_adapter.py` - if just wrapping
4. `completion_consumer.py` - if unused
5. `completion_handler.py` - if unused

**How to check if dead:**
```bash
# Search for imports
rg "from.*executor.processor import" -A 2
rg "from.*tool_adapter import" -A 2
```

### Day 3: Simplify service.py

**Goal:** Reduce from current ~2000 lines to < 1000

**Strategy:**
1. Move helper functions to separate modules
2. Remove unused branches
3. Simplify orchestration logic
4. Direct tool calls instead of adapters?

---

## Phase 3: Fix Remaining Issues (NEXT WEEK) 🐛

### Issue #35: Batched Updates

**Hypothesis:** Redis publishes are buffered somewhere

**Debug:**
1. Add timestamps to every Redis publish
2. Add timestamps to frontend receives
3. Compare - where's the delay?

**Potential fixes:**
- Flush Redis immediately after publish?
- Frontend buffer issue?

### Issue #37: Agent Execution Drops

**Hypothesis:** Tool execution fails silently

**Debug:**
1. Add error logging to every tool call
2. Check if exception swallowed somewhere
3. Test with simple multi-tool agent

**Potential fixes:**
- Better error propagation
- Tool execution timeout handling

### Issue #38: Duplicate Intro

**Hypothesis:** Session state not preserved

**Debug:**
1. Check session creation logic
2. Verify intro only sent on first message
3. Check session lookup by ID

**Potential fixes:**
- Check message count before sending intro
- Store "intro_sent" flag in session

---

## Success Metrics

### Must Have (Block Release)
- ✅ All 6 issues resolved
- ✅ < 3s time to first token
- ✅ < 10s total response time (simple query)
- ✅ 100% tool execution success rate
- ✅ Stop button works reliably

### Should Have (Quality)
- ✅ < 1000 lines in service.py
- ✅ Clear execution flow (< 4 layers)
- ✅ No duplicate code
- ✅ Integration tests for key flows

### Nice to Have (Polish)
- ✅ Performance metrics logged
- ✅ User-friendly error messages
- ✅ Retry logic for transient failures

---

## Testing Checklist

### Manual Testing

**Basic Flow:**
1. Send message "Hello" → expect response
2. Send follow-up "What did I just say?" → expect "Hello"
3. Click stop mid-response → expect immediate stop
4. Refresh page → expect conversation preserved
5. Send new message → expect streaming continues

**Agent Testing:**
1. "Create a simple agent" → expect agent created
2. "Edit the agent to..." → expect edit succeeds
3. "Run the agent" → expect execution completes
4. Check multiple tool calls work

**Edge Cases:**
1. Very long message (> 1000 chars)
2. Rapid successive messages
3. Network disconnect mid-stream
4. Multiple browser tabs same session

---

## Rollback Plan

**If things break:**

1. **Revert timeout change:**
   ```bash
   git revert HEAD  # Revert synchronous change
   ```

2. **Revert to last known good:**
   ```bash
   git reset --hard <commit-before-changes>
   ```

3. **Emergency fix:**
   - Set AGENTGENERATOR_TIMEOUT=600 (back to 10min)
   - Restart services

---

## Communication Plan

### Stakeholders
- Update every EOD on progress
- Share blockers immediately
- Demo working version when ready

### Documentation
- Update ARCHITECTURE.md with findings
- Create TROUBLESHOOTING.md for common issues
- Document removal decisions

---

## Next Steps RIGHT NOW

1. **Fix heartbeat:** 15s → 10s
2. **Manual test all issues:** Create results doc
3. **Trace execution flow:** Map actual path
4. **Identify dead code:** List candidates
5. **Create removal PR:** One big cleanup

Let's get Copilot rock-solid! 💪
