# Manual Test Checklist - Copilot Streaming Issues

**Backend**: http://localhost:8006 ✅ RUNNING
**Frontend**: http://localhost:3000 ✅ RUNNING

## Instructions
Open http://localhost:3000 in your browser and follow these tests:

---

## ✅ Test #1: No Timeout Toast
**Steps:**
1. Open copilot chat
2. Type: `hello`
3. Send message

**Expected:**
- ✅ Response appears within 12 seconds
- ✅ NO "Stream timed out" toast appears

**Actual Result:**
[ ] PASS
[ ] FAIL - describe what happened:

---

## ✅ Test #2: Response Without Refresh
**Steps:**
1. Send message: `say OK`
2. Wait for response
3. DO NOT refresh page

**Expected:**
- ✅ Response appears immediately without page refresh

**Actual Result:**
[ ] PASS
[ ] FAIL - did you need to refresh?

---

## ✅ Test #3: Real-time Streaming
**Steps:**
1. Send message: `count to 10 slowly`
2. Watch how the response appears

**Expected:**
- ✅ Text appears gradually, word by word or phrase by phrase
- ❌ NOT all at once in a batch

**Actual Result:**
[ ] PASS - text streamed gradually
[ ] FAIL - text appeared all at once

---

## ✅ Test #4: Loading State Clears
**Steps:**
1. Send message: `hi`
2. Watch the loading indicator (button state)

**Expected:**
- ✅ Loading indicator appears while processing
- ✅ Loading indicator disappears when done
- ❌ NOT stuck on "red button" or loading state

**Actual Result:**
[ ] PASS
[ ] FAIL - button stuck in loading state

---

## ✅ Test #5: Agent Tools Work (SKIP FOR NOW)
This requires creating an agent which is complex. We'll test this separately.

---

## ✅ Test #6: No Repeated Introduction
**Steps:**
1. Send: `My name is Alice`
2. Wait for response
3. Send: `What's my name?`

**Expected:**
- ✅ Second response remembers "Alice"
- ❌ Second response does NOT show Otto introduction again

**Actual Result:**
[ ] PASS - remembered name
[ ] FAIL - repeated introduction or forgot name

---

## Report Results

After testing, report which tests PASSED or FAILED.

If any fail, also check browser console (F12 → Console) and Network tab (F12 → Network → look for `/stream` requests) for errors.
