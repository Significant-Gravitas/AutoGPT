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

# Read PR description to understand the WHY
gh pr view {N} --json body --jq '.body'

git log --oneline dev..HEAD | head -20
git diff dev --stat
```

Read the PR description and changed files to understand:
0. **Why does this PR exist?** What problem does it solve? (from the `## Why` section)
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

The `claude_agent_sdk` Python package **bundles its own Claude CLI binary** — no need to install `@anthropic-ai/claude-code` via npm. The backend auto-provisions credentials from environment variables on startup.

Run the helper script to extract tokens from your host and auto-update `backend/.env` (works on macOS, Linux, and Windows/WSL):

```bash
# Extracts OAuth tokens and writes CLAUDE_CODE_OAUTH_TOKEN + CLAUDE_CODE_REFRESH_TOKEN into .env
bash $BACKEND_DIR/scripts/refresh_claude_token.sh --env-file $BACKEND_DIR/.env
```

**How it works:** The script reads the OAuth token from:
- **macOS**: system keychain (`"Claude Code-credentials"`)
- **Linux/WSL**: `~/.claude/.credentials.json`
- **Windows**: `%APPDATA%/claude/.credentials.json`

It sets `CLAUDE_CODE_OAUTH_TOKEN`, `CLAUDE_CODE_REFRESH_TOKEN`, and `CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=true` in the `.env` file. On container startup, the backend auto-provisions `~/.claude/.credentials.json` inside the container from these env vars. The SDK's bundled CLI then authenticates using that file. No `claude login`, no npm install needed.

**Note:** The OAuth token expires (~24h). If copilot returns auth errors, re-run the script and restart: `$BACKEND_DIR/scripts/refresh_claude_token.sh --env-file $BACKEND_DIR/.env && docker compose up -d copilot_executor`

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
if [ ${PIPESTATUS[0]} -ne 0 ]; then echo "ERROR: Docker build failed"; exit 1; fi

cd $PLATFORM_DIR && docker compose up -d 2>&1 | tail -20
if [ ${PIPESTATUS[0]} -ne 0 ]; then echo "ERROR: Docker compose up failed"; exit 1; fi
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

## Step 5: Record results and take screenshots

**Take a screenshot at every significant test step** — before and after interactions, on success, and on failure. Name them sequentially with descriptive names:

```bash
agent-browser --session-name pr-test screenshot $RESULTS_DIR/{NN}-{description}.png
# Examples:
# $RESULTS_DIR/01-login-page.png
# $RESULTS_DIR/02-builder-with-block.png
# $RESULTS_DIR/03-copilot-response.png
# $RESULTS_DIR/04-agent-execution-result.png
# $RESULTS_DIR/05-error-state.png
```

**Aim for at least one screenshot per test scenario.** More is better — screenshots are the primary evidence that tests were actually run.

## Step 6: Show results to user with screenshots

**CRITICAL: After all tests complete, you MUST show every screenshot to the user using the Read tool, with an explanation of what each screenshot shows.** This is the most important part of the test report — the user needs to visually verify the results.

For each screenshot:
1. Use the `Read` tool to display the PNG file (Claude can read images)
2. Write a 1-2 sentence explanation below it describing:
   - What page/state is being shown
   - What the screenshot proves (which test scenario it validates)
   - Any notable details visible in the UI

Format the output like this:

```markdown
### Screenshot 1: {descriptive title}
[Read the PNG file here]

**What it shows:** {1-2 sentence explanation of what this screenshot proves}

---
```

After showing all screenshots, output a summary table:

| # | Scenario | Result |
|---|----------|--------|
| 1 | {name} | PASS/FAIL |
| 2 | ... | ... |

**IMPORTANT:** As you show each screenshot and record test results, persist them in shell variables for Step 7:

```bash
# Build these variables during Step 6 — they are required by Step 7's script
declare -A SCREENSHOT_EXPLANATIONS=(
  ["01-login-page.png"]="Shows the login page loaded successfully with SSO options visible."
  ["02-builder-with-block.png"]="The builder canvas displays the newly added block connected to the trigger."
  # ... one entry per screenshot, using the same explanations you showed the user above
)

TEST_RESULTS_TABLE="| 1 | Login flow | PASS |
| 2 | Builder block addition | PASS |
| 3 | Copilot chat | FAIL |"
# ... one row per test scenario with actual results
```

## Step 7: Post test report as PR comment with screenshots

Upload screenshots to the PR using the GitHub Git API (no local git operations — safe for worktrees), then post a comment with inline images and per-screenshot explanations.

```bash
# Upload screenshots via GitHub Git API (creates blobs, tree, commit, and ref remotely)
REPO="Significant-Gravitas/AutoGPT"
SCREENSHOTS_BRANCH="test-screenshots/pr-${PR_NUMBER}"
SCREENSHOTS_DIR="test-screenshots/PR-${PR_NUMBER}"

# Step 1: Create blobs for each screenshot and build tree JSON
TREE_JSON='['
FIRST=true
for img in $RESULTS_DIR/*.png; do
  BASENAME=$(basename "$img")
  B64=$(base64 < "$img")
  BLOB_SHA=$(gh api "repos/${REPO}/git/blobs" -f content="$B64" -f encoding="base64" --jq '.sha')
  if [ "$FIRST" = true ]; then FIRST=false; else TREE_JSON+=','; fi
  TREE_JSON+="{\"path\":\"${SCREENSHOTS_DIR}/${BASENAME}\",\"mode\":\"100644\",\"type\":\"blob\",\"sha\":\"${BLOB_SHA}\"}"
done
TREE_JSON+=']'

# Step 2: Create tree, commit, and branch ref
TREE_SHA=$(echo "$TREE_JSON" | jq -c '{tree: .}' | gh api "repos/${REPO}/git/trees" --input - --jq '.sha')
COMMIT_SHA=$(gh api "repos/${REPO}/git/commits" \
  -f message="test: add E2E test screenshots for PR #${PR_NUMBER}" \
  -f tree="$TREE_SHA" \
  --jq '.sha')
gh api "repos/${REPO}/git/refs" \
  -f ref="refs/heads/${SCREENSHOTS_BRANCH}" \
  -f sha="$COMMIT_SHA" 2>/dev/null \
  || gh api "repos/${REPO}/git/refs/heads/${SCREENSHOTS_BRANCH}" \
    -X PATCH -f sha="$COMMIT_SHA" -f force=true
```

Then post the comment with **inline images AND explanations for each screenshot**:

```bash
REPO_URL="https://raw.githubusercontent.com/${REPO}/${SCREENSHOTS_BRANCH}"

# Build image markdown using SCREENSHOT_EXPLANATIONS and TEST_RESULTS_TABLE from Step 6

IMAGE_MARKDOWN=""
for img in $RESULTS_DIR/*.png; do
  BASENAME=$(basename "$img")
  TITLE=$(echo "${BASENAME%.png}" | sed 's/^[0-9]*-//' | sed 's/-/ /g' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) tolower(substr($i,2))}1')
  EXPLANATION="${SCREENSHOT_EXPLANATIONS[$BASENAME]}"
  IMAGE_MARKDOWN="${IMAGE_MARKDOWN}
### ${TITLE}
![${BASENAME}](${REPO_URL}/${SCREENSHOTS_DIR}/${BASENAME})
${EXPLANATION}
"
done

# Write comment body to file to avoid shell interpretation issues with special characters
COMMENT_FILE=$(mktemp)
cat > "$COMMENT_FILE" <<INNEREOF
## 🧪 E2E Test Report

| # | Scenario | Result |
|---|----------|--------|
${TEST_RESULTS_TABLE}

${IMAGE_MARKDOWN}
INNEREOF

gh api "repos/${REPO}/issues/$PR_NUMBER/comments" -F body=@"$COMMENT_FILE"
rm -f "$COMMENT_FILE"
```

**The PR comment MUST include:**
1. A summary table of all scenarios with PASS/FAIL
2. Every screenshot rendered inline (not just linked)
3. A 1-2 sentence explanation below each screenshot describing what it proves

This approach uses the GitHub Git API to create blobs, trees, commits, and refs entirely server-side. No local `git checkout` or `git push` — safe for worktrees and won't interfere with the PR branch.

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

### Problem: Copilot returns auth errors in subscription mode
**Cause:** `CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=true` but `CLAUDE_CODE_OAUTH_TOKEN` is not set or expired.
**Fix:** Re-extract the OAuth token from macOS keychain (see step 3b, Option 1) and recreate the container (`docker compose up -d copilot_executor`). The backend auto-provisions `~/.claude/.credentials.json` from the env var on startup. No `npm install` or `claude login` needed — the SDK bundles its own CLI binary.

### Problem: agent-browser can't find chromium
**Cause:** The Dockerfile auto-provisions system chromium on all architectures (including ARM64). If your branch is behind `dev`, this may not be present yet.
**Fix:** Check if chromium exists: `which chromium || which chromium-browser`. If missing, install it: `apt-get install -y chromium` and set `AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium` in the container environment.

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
