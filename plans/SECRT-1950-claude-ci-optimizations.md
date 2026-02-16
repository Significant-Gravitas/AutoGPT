# Implementation Plan: SECRT-1950 - Apply E2E CI Optimizations to Claude Code Workflows

## Ticket
[SECRT-1950](https://linear.app/autogpt/issue/SECRT-1950)

## Summary
Apply Pwuts's CI performance optimizations from PR #12090 to Claude Code workflows.

## Reference PR
https://github.com/Significant-Gravitas/AutoGPT/pull/12090

---

## Analysis

### Current State (claude.yml)

**pnpm caching (lines 104-118):**
```yaml
- name: Set up Node.js
  uses: actions/setup-node@v6
  with:
    node-version: "22"

- name: Enable corepack
  run: corepack enable

- name: Set pnpm store directory
  run: |
    pnpm config set store-dir ~/.pnpm-store
    echo "PNPM_HOME=$HOME/.pnpm-store" >> $GITHUB_ENV

- name: Cache frontend dependencies
  uses: actions/cache@v5
  with:
    path: ~/.pnpm-store
    key: ${{ runner.os }}-pnpm-${{ hashFiles('autogpt_platform/frontend/pnpm-lock.yaml', 'autogpt_platform/frontend/package.json') }}
    restore-keys: |
      ${{ runner.os }}-pnpm-${{ hashFiles('autogpt_platform/frontend/pnpm-lock.yaml') }}
      ${{ runner.os }}-pnpm-
```

**Docker setup (lines 134-165):**
- Uses `docker-buildx-action@v3` 
- Has manual Docker image caching via `actions/cache`
- Runs `docker compose up` without buildx bake optimization

### Pwuts's Optimizations (PR #12090)

1. **Simplified pnpm caching** - Use `setup-node` built-in cache:
```yaml
- name: Enable corepack
  run: corepack enable

- name: Set up Node
  uses: actions/setup-node@v6
  with:
    node-version: "22.18.0"
    cache: "pnpm"
    cache-dependency-path: autogpt_platform/frontend/pnpm-lock.yaml
```

2. **Docker build caching via buildx bake**:
```yaml
- name: Set up Docker Buildx
  uses: docker/setup-buildx-action@v3
  with:
    driver: docker-container
    driver-opts: network=host

- name: Expose GHA cache to docker buildx CLI
  uses: crazy-max/ghaction-github-runtime@v3

- name: Build Docker images (with cache)
  run: |
    pip install pyyaml
    docker compose -f docker-compose.yml config > docker-compose.resolved.yml
    python ../.github/workflows/scripts/docker-ci-fix-compose-build-cache.py \
      --source docker-compose.resolved.yml \
      --cache-from "type=gha" \
      --cache-to "type=gha,mode=max" \
      ...
    docker buildx bake --allow=fs.read=.. -f docker-compose.resolved.yml --load
```

---

## Proposed Changes

### 1. Update pnpm caching in `claude.yml`

**Before:**
- Manual cache key generation
- Separate `actions/cache` step
- Manual pnpm store directory config

**After:**
- Use `setup-node` built-in `cache: "pnpm"` option
- Remove manual cache step
- Keep `corepack enable` before `setup-node`

### 2. Update Docker build in `claude.yml`

**Before:**
- Manual Docker layer caching via `actions/cache` with `/tmp/.buildx-cache`
- Simple `docker compose build`

**After:**
- Use `crazy-max/ghaction-github-runtime@v3` to expose GHA cache
- Use `docker-ci-fix-compose-build-cache.py` script
- Build with `docker buildx bake`

### 3. Apply same changes to other Claude workflows

- `claude-dependabot.yml` - Check if it has similar patterns
- `claude-ci-failure-auto-fix.yml` - Check if it has similar patterns
- `copilot-setup-steps.yml` - Reusable workflow, may be the source of truth

---

## Files to Modify

1. `.github/workflows/claude.yml`
2. `.github/workflows/claude-dependabot.yml` (if applicable)
3. `.github/workflows/claude-ci-failure-auto-fix.yml` (if applicable)

## Dependencies

- PR #12090 must be merged first (provides the `docker-ci-fix-compose-build-cache.py` script)
- Backend Dockerfile optimizations (already in PR #12090)

---

## Test Plan

1. Create PR with changes
2. Trigger Claude workflow manually or via `@claude` mention on a test issue
3. Compare CI runtime before/after
4. Verify Claude agent still works correctly (can checkout, build, run tests)

---

## Risk Assessment

**Low risk:**
- These are CI infrastructure changes, not code changes
- If caching fails, builds fall back to uncached (slower but works)
- Changes mirror proven patterns from PR #12090

---

## Questions for Reviewer

1. Should we wait for PR #12090 to merge before creating this PR?
2. Does `copilot-setup-steps.yml` need updating, or is it a separate concern?
3. Any concerns about cache key collisions between frontend E2E and Claude workflows?

---

## Verified

- ✅ **`claude-dependabot.yml`**: Has same pnpm caching pattern as `claude.yml` (manual `actions/cache`) — NEEDS UPDATE
- ✅ **`claude-ci-failure-auto-fix.yml`**: Simple workflow with no pnpm or Docker caching — NO CHANGES NEEDED
- ✅ **Script path**: `docker-ci-fix-compose-build-cache.py` will be at `.github/workflows/scripts/` after PR #12090 merges
- ✅ **Test seed caching**: NOT APPLICABLE — Claude workflows spin up a dev environment but don't run E2E tests with pre-seeded data. The seed caching in PR #12090 is specific to the frontend E2E test suite which needs consistent test data. Claude just needs the services running.
