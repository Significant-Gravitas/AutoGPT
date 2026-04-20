# E2E Test Report: PR #12699 — feat(frontend/builder): add chat panel for interactive agent editing
Date: 2026-04-08
Branch: feat/builder-chat-panel

## Environment
- Docker services: frontend, rest_server, executor, copilot_executor, websocket_server, supabase, redis, rabbitmq
- Copilot: OpenRouter API key mode
- Test user: test@test.com
- Test graph: Math Solver Agent (4cf4fef2-cedf-4776-858e-dab37a9984a3)

## Test Results

### Scenario 1: Chat panel opens on click
**Steps:**
1. Navigate to `/build?flowID=4cf4fef2-cedf-4776-858e-dab37a9984a3`
2. Click the floating chat button (bottom-right)
**Expected:** Panel opens with "Chat with Builder" header
**Actual:** Panel opens correctly with header visible
**Result:** PASS

### Scenario 2: Session creation and seed message
**Steps:**
1. After opening panel, wait for session creation
2. Observe seed message sent automatically with graph context
**Expected:** Seed message includes graph structure + JSON format instructions
**Actual:** Seed message sent with 4 nodes, 5 connections, and full JSON format instructions
**Result:** PASS

### Scenario 3: AI response visible (auto-scroll fix)
**Steps:**
1. Wait ~25 seconds after opening panel
2. Observe chat panel
**Expected:** AI response visible in panel (not hidden below fold)
**Actual:** AI response "This agent is a natural language math calculator..." visible
**Result:** PASS (fix: added `useEffect` scrolling to `messagesEndRef` on `messages.length` change)

### Scenario 4: AI outputs correct JSON format
**Steps:**
1. Ask AI to update OrchestratorBlock's system_prompt field
2. Wait for AI response
**Expected:** AI outputs `{"action": "update_node_input", "node_id": "...", "key": "...", "value": "..."}`
**Actual:** AI correctly output `{"action": "update_node_input", "node_id": "2ba71bf5-9a21-4440-a308-7a5838d34176", "key": "system_prompt", "value": "You are a helpful math assistant..."}`
**Result:** PASS (fix: strengthened seed instruction with "MUST output ... EXACTLY these formats — no other structure is recognized")

### Scenario 5: "AI applied these changes" panel appears
**Steps:**
1. After AI responds with graph modification
2. Observe chat panel
**Expected:** "AI applied these changes" section with "Applied" button
**Actual:** "AI applied these changes" label visible; "Applied" button shown in applied (disabled) state
**Result:** PASS

### Scenario 6: Canvas auto-refresh after AI modifies graph
**Steps:**
1. After AI responds, check backend logs
**Expected:** Graph re-fetched via `invalidateQueries`
**Actual:** Backend logs confirm `GET /api/graphs/4cf4fef2-...` fired after AI response completion
**Result:** PASS (implemented via `prevStatusRef` tracking "streaming"→"ready" transition)

### Scenario 7: Chat panel stays open during interaction
**Steps:**
1. Open panel, wait for AI response, send user message, get second AI response
**Expected:** Panel remains open throughout
**Actual:** Panel remained open for all interactions
**Result:** PASS

## Summary
- Total: 7 scenarios
- Passed: 7
- Failed: 0

## Fixes in this PR (beyond original feature)
1. `ffa955044d` — Strengthened JSON format instruction so AI reliably outputs `{"action":...}` format
2. `109f28d9d1` — Auto-scroll to bottom when AI responds (seed message was too long, pushing AI reply off-screen)
3. `0999739d19` — Canvas auto-refresh via `invalidateQueries` after AI edits graph server-side
