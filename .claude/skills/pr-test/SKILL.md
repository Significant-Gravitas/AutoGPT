---
name: pr-test
description: "E2E manual testing of PRs/branches using docker compose, agent-browser, and API calls. TRIGGER when user asks to manually test a PR, test a feature end-to-end, or run integration tests against a running system."
user-invocable: true
argument-hint: "[worktree path or PR number] — tests the PR in the given worktree. Optional flags: --fix (auto-fix issues found)"
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Manual E2E Test

Test a PR/branch end-to-end by building the full platform, interacting via browser and API, capturing screenshots, and reporting results.

## Arguments

- `$ARGUMENTS` — worktree path (e.g. `$REPO_ROOT`) or PR number
- If `--fix` flag is present, auto-fix bugs found and push fixes (like pr-address loop)

## Step 0: Resolve the target

```bash
# If argument is a PR number, find its worktree
gh pr view {N} --json headRefName --jq '.headRefName'
# If argument is a path, use it directly
```

Determine:
- `REPO_ROOT` — the root repo directory: `git -C "$WORKTREE_PATH" worktree list | head -1 | awk '{print $1}'` (or `git rev-parse --show-toplevel` if not a worktree)
- `WORKTREE_PATH` — the worktree directory
- `PLATFORM_DIR` — `$WORKTREE_PATH/autogpt_platform`
- `BACKEND_DIR` — `$PLATFORM_DIR/backend`
- `FRONTEND_DIR` — `$PLATFORM_DIR/frontend`
- `PR_NUMBER` — the PR number (from `gh pr list --head $(git branch --show-current)`)
- `PR_TITLE` — the PR title, slugified (e.g. "Add copilot permissions" → "add-copilot-permissions")
- `RESULTS_DIR` — `$REPO_ROOT/test-results/PR-{PR_NUMBER}-{slugified-title}`

Create the results directory:
```bash
PR_NUMBER=$(cd $WORKTREE_PATH && gh pr list --head $(git branch --show-current) --repo Significant-Gravitas/AutoGPT --json number --jq '.[0].number')
PR_TITLE=$(cd $WORKTREE_PATH && gh pr list --head $(git branch --show-current) --repo Significant-Gravitas/AutoGPT --json title --jq '.[0].title' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-//;s/-$//' | head -c 50)
RESULTS_DIR="$REPO_ROOT/test-results/PR-${PR_NUMBER}-${PR_TITLE}"
mkdir -p $RESULTS_DIR
```

**Test user credentials** (for logging into the UI or verifying results manually):
- Email: `test@test.com`
- Password: `testtest123`

## Step 1: Understand the PR

Before testing, understand what changed:

```bash
cd $WORKTREE_PATH
git log --oneline dev..HEAD | head -20
git diff dev --stat
```

Read the changed files to understand:
1. What feature/fix does this PR implement?
2. What components are affected? (backend, frontend, copilot, executor, etc.)
3. What are the key user-facing behaviors to test?

## Step 2: Write test scenarios

Based on the PR analysis, write a test plan to `$RESULTS_DIR/test-plan.md`:

```markdown
# Test Plan: PR #{N} — {title}

## Scenarios
1. [Scenario name] — [what to verify]
2. ...

## API Tests (if applicable)
1. [Endpoint] — [expected behavior]

## UI Tests (if applicable)
1. [Page/component] — [interaction to test]

## Negative Tests
1. [What should NOT happen]
```

**Be critical** — include edge cases, error paths, and security checks.

## Step 3: Environment setup

### 3a. Copy .env files from the root worktree

The root worktree (`$REPO_ROOT`) has the canonical `.env` files with all API keys. Copy them to the target worktree:

```bash
# CRITICAL: .env files are NOT checked into git. They must be copied manually.
cp $REPO_ROOT/autogpt_platform/.env $PLATFORM_DIR/.env
cp $REPO_ROOT/autogpt_platform/backend/.env $BACKEND_DIR/.env
cp $REPO_ROOT/autogpt_platform/frontend/.env $FRONTEND_DIR/.env
```

### 3b. Configure copilot authentication

The copilot needs an LLM API to function. Two approaches (try subscription first):

#### Option 1: Subscription mode (preferred — uses your Claude Max/Pro subscription)

On macOS, Claude Code stores OAuth credentials in the **system keychain**. Extract the token and pass it to Docker:

```bash
# Extract OAuth access token from macOS keychain
CLAUDE_OAUTH_TOKEN=$(security find-generic-password -s "Claude Code-credentials" -w 2>/dev/null | jq -r '.claudeAiOauth.accessToken // ""' 2>/dev/null)

if [ -n "$CLAUDE_OAUTH_TOKEN" ]; then
  echo "Found Claude OAuth token from keychain"
  # Pass it as CLAUDE_CODE_OAUTH_TOKEN env var to copilot_executor
  # Add to docker-compose.override.yml or the backend .env
  grep -q "^CLAUDE_CODE_OAUTH_TOKEN=" $BACKEND_DIR/.env && \
    perl -i -pe "s|^CLAUDE_CODE_OAUTH_TOKEN=.*|CLAUDE_CODE_OAUTH_TOKEN=$CLAUDE_OAUTH_TOKEN|" $BACKEND_DIR/.env || \
    echo "CLAUDE_CODE_OAUTH_TOKEN=$CLAUDE_OAUTH_TOKEN" >> $BACKEND_DIR/.env
  # Keep subscription mode enabled
  perl -i -pe 's/CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=false/CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=true/' $BACKEND_DIR/.env 2>/dev/null
else
  echo "No Claude OAuth token found — falling back to OpenRouter API key mode"
fi
```

**Note:** The OAuth token expires (~24h). If copilot returns auth errors, re-extract from keychain. On Linux, credentials are stored in `~/.claude/.credentials.json` (plaintext fallback) — read the `claudeAiOauth.accessToken` from there instead.

**Prerequisite:** You must have run `claude login` on the host machine at least once (which sets up the keychain entry).

**Claude CLI in container:** Subscription mode requires the `claude` CLI inside the copilot executor container. After the container starts, install it at runtime:
```bash
docker exec autogpt_platform-copilot_executor-1 which claude 2>/dev/null || \
  docker exec autogpt_platform-copilot_executor-1 npm install -g @anthropic-ai/claude-code 2>&1 | tail -3
```
This is lost on container rebuild — re-run after each `docker compose up --build`.

#### Option 2: OpenRouter API key mode (fallback)

If subscription mode doesn't work, switch to API key mode using OpenRouter:

```bash
# In $BACKEND_DIR/.env, ensure these are set:
CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=false
CHAT_API_KEY=<value of OPEN_ROUTER_API_KEY from the same .env>
CHAT_BASE_URL=https://openrouter.ai/api/v1
CHAT_USE_CLAUDE_AGENT_SDK=true
```

Use `sed` to update these values:
```bash
ORKEY=$(grep "^OPEN_ROUTER_API_KEY=" $BACKEND_DIR/.env | cut -d= -f2)
[ -n "$ORKEY" ] || { echo "ERROR: OPEN_ROUTER_API_KEY is missing in $BACKEND_DIR/.env"; exit 1; }
perl -i -pe 's/CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=true/CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=false/' $BACKEND_DIR/.env
# Add or update CHAT_API_KEY and CHAT_BASE_URL
grep -q "^CHAT_API_KEY=" $BACKEND_DIR/.env && perl -i -pe "s|^CHAT_API_KEY=.*|CHAT_API_KEY=$ORKEY|" $BACKEND_DIR/.env || echo "CHAT_API_KEY=$ORKEY" >> $BACKEND_DIR/.env
grep -q "^CHAT_BASE_URL=" $BACKEND_DIR/.env && perl -i -pe 's|^CHAT_BASE_URL=.*|CHAT_BASE_URL=https://openrouter.ai/api/v1|' $BACKEND_DIR/.env || echo "CHAT_BASE_URL=https://openrouter.ai/api/v1" >> $BACKEND_DIR/.env
```

### 3c. Stop conflicting containers

```bash
# Stop any running app containers (keep infra: supabase, redis, rabbitmq, clamav)
docker ps --format "{{.Names}}" | grep -E "rest_server|executor|copilot|websocket|database_manager|scheduler|notification|frontend|migrate" | while read name; do
  docker stop "$name" 2>/dev/null
done
```

### 3e. Build and start

```bash
cd $PLATFORM_DIR && docker compose build --no-cache 2>&1 | tail -20
cd $PLATFORM_DIR && docker compose up -d 2>&1 | tail -20
```

**Note:** If the container appears to be running old code (e.g. missing PR changes), use `docker compose build --no-cache` to force a full rebuild. Docker BuildKit may sometimes reuse cached `COPY` layers from a previous build on a different branch.

**Expected time: 3-8 minutes** for build, 5-10 minutes with `--no-cache`.

### 3f. Wait for services to be ready

```bash
# Poll until backend and frontend respond
for i in $(seq 1 60); do
  BACKEND=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8006/docs 2>/dev/null)
  FRONTEND=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 2>/dev/null)
  if [ "$BACKEND" = "200" ] && [ "$FRONTEND" = "200" ]; then
    echo "Services ready"
    break
  fi
  sleep 5
done
```


### 3h. Create test user and get auth token

```bash
ANON_KEY=$(grep "NEXT_PUBLIC_SUPABASE_ANON_KEY=" $FRONTEND_DIR/.env | sed 's/.*NEXT_PUBLIC_SUPABASE_ANON_KEY=//' | tr -d '[:space:]')

# Signup (idempotent — returns "User already registered" if exists)
RESULT=$(curl -s -X POST 'http://localhost:8000/auth/v1/signup' \
  -H "apikey: $ANON_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"email":"test@test.com","password":"testtest123"}')

# If "Database error finding user", restart supabase-auth and retry
if echo "$RESULT" | grep -q "Database error"; then
  docker restart supabase-auth && sleep 5
  curl -s -X POST 'http://localhost:8000/auth/v1/signup' \
    -H "apikey: $ANON_KEY" \
    -H 'Content-Type: application/json' \
    -d '{"email":"test@test.com","password":"testtest123"}'
fi

# Get auth token
TOKEN=$(curl -s -X POST 'http://localhost:8000/auth/v1/token?grant_type=password' \
  -H "apikey: $ANON_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"email":"test@test.com","password":"testtest123"}' | jq -r '.access_token // ""')
```

**Use this token for ALL API calls:**
```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8006/api/...
```

## Step 4: Run tests

### Service ports reference

| Service | Port | URL |
|---------|------|-----|
| Frontend | 3000 | http://localhost:3000 |
| Backend REST | 8006 | http://localhost:8006 |
| Supabase Auth (via Kong) | 8000 | http://localhost:8000 |
| Executor | 8002 | http://localhost:8002 |
| Copilot Executor | 8008 | http://localhost:8008 |
| WebSocket | 8001 | http://localhost:8001 |
| Database Manager | 8005 | http://localhost:8005 |
| Redis | 6379 | localhost:6379 |
| RabbitMQ | 5672 | localhost:5672 |

### API testing

Use `curl` with the auth token for backend API tests:

```bash
# Example: List agents
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8006/api/graphs | jq . | head -20

# Example: Create an agent
curl -s -X POST http://localhost:8006/api/graphs \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{...}' | jq .

# Example: Run an agent
curl -s -X POST "http://localhost:8006/api/graphs/{graph_id}/execute" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"data": {...}}'

# Example: Get execution results
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8006/api/graphs/{graph_id}/executions/{exec_id}" | jq .
```

### Browser testing with agent-browser

```bash
# Close any existing session
agent-browser close 2>/dev/null || true

# Use --session-name to persist cookies across navigations
# This means login only needs to happen once per test session
agent-browser --session-name pr-test open 'http://localhost:3000/login' --timeout 15000

# Get interactive elements
agent-browser --session-name pr-test snapshot | grep "textbox\|button"

# Login
agent-browser --session-name pr-test fill {email_ref} "test@test.com"
agent-browser --session-name pr-test fill {password_ref} "testtest123"
agent-browser --session-name pr-test click {login_button_ref}
sleep 5

# Dismiss cookie banner if present
agent-browser --session-name pr-test click 'text=Accept All' 2>/dev/null || true

# Navigate — cookies are preserved so login persists
agent-browser --session-name pr-test open 'http://localhost:3000/copilot' --timeout 10000

# Take screenshot
agent-browser --session-name pr-test screenshot $RESULTS_DIR/01-page.png

# Interact with elements
agent-browser --session-name pr-test fill {ref} "text"
agent-browser --session-name pr-test press "Enter"
agent-browser --session-name pr-test click {ref}
agent-browser --session-name pr-test click 'text=Button Text'

# Read page content
agent-browser --session-name pr-test snapshot | grep "text:"
```

**Key pages:**
- `/copilot` — CoPilot chat (for testing copilot features)
- `/build` — Agent builder (for testing block/node features)
- `/build?flowID={id}` — Specific agent in builder
- `/library` — Agent library (for testing listing/import features)
- `/library/agents/{id}` — Agent detail with run history
- `/marketplace` — Marketplace

### Checking logs

```bash
# Backend REST server
docker logs autogpt_platform-rest_server-1 2>&1 | tail -30

# Executor (runs agent graphs)
docker logs autogpt_platform-executor-1 2>&1 | tail -30

# Copilot executor (runs copilot chat sessions)
docker logs autogpt_platform-copilot_executor-1 2>&1 | tail -30

# Frontend
docker logs autogpt_platform-frontend-1 2>&1 | tail -30

# Filter for errors
docker logs autogpt_platform-executor-1 2>&1 | grep -i "error\|exception\|traceback" | tail -20
```

### Copilot chat testing

The copilot uses SSE streaming. To test via API:

```bash
# Create a session
SESSION_ID=$(curl -s -X POST 'http://localhost:8006/api/chat/sessions' \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{}' | jq -r '.id // .session_id // ""')

# Stream a message (SSE - will stream chunks)
curl -N -X POST "http://localhost:8006/api/chat/sessions/$SESSION_ID/stream" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"message": "Hello, what can you help me with?"}' \
  --max-time 60 2>/dev/null | head -50
```

Or test via browser (preferred for UI verification):
```bash
agent-browser --session-name pr-test open 'http://localhost:3000/copilot' --timeout 10000
# ... fill chat input and press Enter, wait 20-30s for response
```

## Step 5: Record results

For each test scenario, record in `$RESULTS_DIR/test-report.md`:

```markdown
# E2E Test Report: PR #{N} — {title}
Date: {date}
Branch: {branch}
Worktree: {path}

## Environment
- Docker services: [list running containers]
- API keys: OpenRouter={present/missing}, E2B={present/missing}

## Test Results

### Scenario 1: {name}
**Steps:**
1. ...
2. ...
**Expected:** ...
**Actual:** ...
**Result:** PASS / FAIL
**Screenshot:** {filename}.png
**Logs:** (if relevant)

### Scenario 2: {name}
...

## Summary
- Total: X scenarios
- Passed: Y
- Failed: Z
- Bugs found: [list]
```

Take screenshots at each significant step:
```bash
agent-browser --session-name pr-test screenshot $RESULTS_DIR/{NN}-{description}.png
```

## Step 6: Report results

After all tests complete, output a summary to the user:

1. Table of all scenarios with PASS/FAIL
2. Screenshots of failures (read the PNG files to show them)
3. Any bugs found with details
4. Recommendations

### Post test results as PR comment with screenshots

Upload screenshots to the PR and post a comment with the results. GitHub PR comments support images via drag-and-drop upload URLs.

```bash
# Upload each screenshot and collect markdown image links
IMAGES=""
for img in $RESULTS_DIR/*.png; do
  BASENAME=$(basename "$img")
  # Upload to GitHub via the repo's issue attachment API
  UPLOAD_URL=$(gh api repos/Significant-Gravitas/AutoGPT/issues/$PR_NUMBER/comments \
    --method POST \
    -f body="![${BASENAME}](https://github.com/user-attachments/placeholder)" 2>/dev/null | jq -r '.id' 2>/dev/null)
  # Since GitHub doesn't have a direct image upload API, use gh CLI to attach
  IMAGES="$IMAGES\n![${BASENAME}]($img)"
done

# Post the test report as a PR comment with embedded screenshots
# Upload screenshots first by creating a temporary gist or using repo assets
cd $WORKTREE_PATH

# Copy screenshots into a branch and push, then reference them
SCREENSHOTS_BRANCH="test-screenshots/pr-${PR_NUMBER}"
git checkout -b "$SCREENSHOTS_BRANCH" 2>/dev/null || git checkout "$SCREENSHOTS_BRANCH"
mkdir -p "test-screenshots/PR-${PR_NUMBER}"
cp $RESULTS_DIR/*.png "test-screenshots/PR-${PR_NUMBER}/"
git add "test-screenshots/PR-${PR_NUMBER}/"
git commit -m "test: add E2E test screenshots for PR #${PR_NUMBER}" --allow-empty
git push origin "$SCREENSHOTS_BRANCH" --force
git checkout -  # go back to original branch

# Build image URLs from the pushed branch
REPO_URL="https://raw.githubusercontent.com/Significant-Gravitas/AutoGPT/${SCREENSHOTS_BRANCH}"
IMAGE_MARKDOWN=""
for img in $RESULTS_DIR/*.png; do
  BASENAME=$(basename "$img")
  IMAGE_MARKDOWN="$IMAGE_MARKDOWN
![${BASENAME}](${REPO_URL}/test-screenshots/PR-${PR_NUMBER}/${BASENAME})"
done

# Post the comment
gh api repos/Significant-Gravitas/AutoGPT/issues/$PR_NUMBER/comments -f body="$(cat <<EOF
## 🧪 E2E Test Report

$(cat $RESULTS_DIR/test-report.md)

### Screenshots
${IMAGE_MARKDOWN}
EOF
)"
```

**Alternative (simpler):** If you don't want to push screenshots to a branch, just post the text report without images:

```bash
gh api repos/Significant-Gravitas/AutoGPT/issues/$PR_NUMBER/comments -f body="$(cat <<EOF
## 🧪 E2E Test Report

$(cat $RESULTS_DIR/test-report.md)

_Screenshots saved locally at: $RESULTS_DIR/_
EOF
)"
```

The first approach pushes screenshots to a temporary branch so they render inline in the PR comment. The second just references the local path. **Use the first approach when screenshots are important for review.**

## Fix mode (--fix flag)

When `--fix` is present, after finding a bug:

1. Identify the root cause in the code
2. Fix it in the worktree
3. Rebuild the affected service: `cd $PLATFORM_DIR && docker compose up --build -d {service_name}`
4. Re-test the scenario
5. If fix works, commit and push:
   ```bash
   cd $WORKTREE_PATH
   git add -A
   git commit -m "fix: {description of fix}"
   git push
   ```
6. Continue testing remaining scenarios
7. After all fixes, run the full test suite again to ensure no regressions

### Fix loop (like pr-address)

```text
test scenario → find bug → fix code → rebuild service → re-test
→ repeat until all scenarios pass
→ commit + push all fixes
→ run full re-test to verify
```

## Known issues and workarounds

### Problem: "Database error finding user" on signup
**Cause:** Supabase auth service schema cache is stale after migration.
**Fix:** `docker restart supabase-auth && sleep 5` then retry signup.

### Problem: "Claude Code CLI not found" in copilot executor
**Cause:** The Dockerfile doesn't include `@anthropic-ai/claude-code` in npm install.
**Fix:** Either add it to the Dockerfile (`npm install -g agent-browser @anthropic-ai/claude-code`) or install it at runtime: `docker exec autogpt_platform-copilot_executor-1 npm install -g @anthropic-ai/claude-code`

### Problem: Copilot returns 401 "invalid bearer token" from Anthropic
**Cause:** `CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=true` but the container has no OAuth token.
**Fix:** Extract the OAuth token from macOS keychain and pass as `CLAUDE_CODE_OAUTH_TOKEN` env var (see step 3b, Option 1). Or switch to OpenRouter API key mode (Option 2).

### Problem: Docker build fails on ARM64 with chromium errors
**Cause:** `Chrome for Testing` has no ARM64 binary. Dockerfile uses `TARGETARCH` conditional that fails.
**Fix:** This is fixed by PR #12473 (merged to `dev`). If your branch is behind `dev`, merge `dev` into it. If still unfixed, replace with unconditional `apt-get install chromium` + `ENV AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium`.

### Problem: agent-browser selector matches multiple elements
**Cause:** `text=X` matches all elements containing that text.
**Fix:** Use `agent-browser snapshot` to get specific `ref=eNN` references, then use those: `agent-browser click eNN`.

### Problem: Frontend shows cookie banner blocking interaction
**Fix:** `agent-browser click 'text=Accept All'` before other interactions.

### Problem: Container loses npm packages after rebuild
**Cause:** `docker compose up --build` rebuilds the image, losing runtime installs.
**Fix:** Add packages to the Dockerfile instead of installing at runtime.

### Problem: Services not starting after `docker compose up`
**Fix:** Wait and check health: `docker compose ps`. Common cause: migration hasn't finished. Check: `docker logs autogpt_platform-migrate-1 2>&1 | tail -5`. If supabase-db isn't healthy: `docker restart supabase-db && sleep 10`.

### Problem: Docker uses cached layers with old code (PR changes not visible)
**Cause:** `docker compose up --build` reuses cached `COPY` layers from previous builds. If the PR branch changes Python files but the previous build already cached that layer from `dev`, the container runs `dev` code.
**Fix:** Always use `docker compose build --no-cache` for the first build of a PR branch. Subsequent rebuilds within the same branch can use `--build`.

### Problem: `agent-browser open` loses login session
**Cause:** Without session persistence, `agent-browser open` starts fresh.
**Fix:** Use `--session-name pr-test` on ALL agent-browser commands. This auto-saves/restores cookies and localStorage across navigations. Alternatively, use `agent-browser eval "window.location.href = '...'"` to navigate within the same context.

### Problem: Supabase auth returns "Database error querying schema"
**Cause:** The database schema changed (migration ran) but supabase-auth has a stale schema cache.
**Fix:** `docker restart supabase-db && sleep 10 && docker restart supabase-auth && sleep 8`. If user data was lost, re-signup.
