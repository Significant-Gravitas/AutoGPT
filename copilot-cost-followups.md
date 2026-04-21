# Copilot cost-reduction follow-ups

Four independent tasks that reduce per-turn cost for the copilot feature. Each is self-contained — an agent can pick any one and execute without reading the others.

## Shared context

- **Repo:** `Significant-Gravitas/AutoGPT`. Worktree at `/Users/majdyz/Code/AutoGPT1` (branch `feat/copilot-cost-based-rate-limit`, PR #12864) introduces real-cost rate limiting. These follow-ups are separate PRs on top.
- **Commit author:** must be `majdyz <zamil.majdy@agpt.co>` (CLA). Use `GIT_AUTHOR_NAME="majdyz" GIT_AUTHOR_EMAIL="zamil.majdy@agpt.co" GIT_COMMITTER_NAME="majdyz" GIT_COMMITTER_EMAIL="zamil.majdy@agpt.co"`.
- **Commit style:** conventional (`feat(copilot): …`, `perf(copilot): …`, `refactor(copilot): …`).
- **Commit hook bypass:** use `--no-verify` — `poetry run format` + `pnpm format/types` already cover lint. The `export-api-schema` pre-commit hook is slow/flaky.
- **Baseline evidence** (two `copilot:Baseline` vs `copilot:SDK` turns with the message "hello" on 2026-04-21):

  | Path | In / Out | Cache R/W | Cost | Root cause |
  |---|---|---|---|---|
  | SDK | 3 / 81 | 0 / 33.8K | $0.2131 | Turn 1 cache-write (expected) + resume workaround disables preset on turn 2+ |
  | Baseline | 18.7K / 121 | – | $0.0579 | No `cache_control` on any request; 4.3K wasted supplement; 5K-token Langfuse prompt |

  Token breakdown verified via `Anthropic.messages.count_tokens(claude-sonnet-4-5, …)` to within 0.2%.

---

## Task 1 — Add `cache_control` markers to the baseline OpenRouter path

**Why:** `backend/copilot/baseline/service.py` currently sends no `cache_control` hints, so every hello rings up the full 18.7K tokens uncached (~$0.058). OpenRouter supports Anthropic ephemeral caching when the Anthropic-compatible fields are passed through. Marking the system-prompt and last tool schema as ephemeral turns repeat hellos into cache READs at ~10% of the write price.

**Target files / lines:**
- [`autogpt_platform/backend/backend/copilot/baseline/service.py:354-371`](autogpt_platform/backend/backend/copilot/baseline/service.py#L354-L371) — where `_baseline_llm_caller` builds the `client.chat.completions.create(...)` call.

**Approach:**
1. Convert `messages[0]` (the system message) from a plain string into the Anthropic content-blocks form `[{"type": "text", "text": <system_prompt>, "cache_control": {"type": "ephemeral"}}]`.
2. Attach `cache_control={"type":"ephemeral"}` to the LAST tool in the `tools` array (Anthropic caches prefix through the breakpoint, so marking the last tool caches everything above it).
3. Keep `extra_body={"usage": {"include": True}}` unchanged — we still need real cost back.
4. Verify the payload type annotations — `ChatCompletionMessageParam` and `ChatCompletionToolParam` may need adjustment or a `cast` to accept the block form.
5. Add a unit test in `baseline/service_unit_test.py` that asserts `cache_control` is present on both the system block and the last tool schema.

**Measurement:** after the change, send "hello" twice in under 5 minutes. First turn should report ~18.7K uncached input; second should show ~18K `cacheReadTokens` and ~0 uncached. Cost: first ~$0.058, second ~$0.006 (10×).

**Risks:**
- OpenRouter may need explicit header or SDK flag to enable Anthropic-passthrough caching — research `extra_body` options or headers (`anthropic-beta: prompt-caching-2024-07-31` if needed).
- Non-Anthropic models routed through OpenRouter will ignore the markers — safe default, no regression.

**Expected cost impact:** ~90% drop on repeat baseline turns within 5-minute cache TTL.

### Agent prompt (Task 1)

```
You are adding OpenRouter-compatible prompt caching to the copilot baseline path in PR branch `feat/copilot-cost-based-rate-limit` (branch sibling to, or rebased on, PR #12864). Work from `/Users/majdyz/Code/AutoGPT1`. Commit author: `majdyz <zamil.majdy@agpt.co>`. Use `--no-verify` on commits. See `/Users/majdyz/Code/AutoGPT1/copilot-cost-followups.md` Task 1 for background and approach.

Scope:
1. In `backend/copilot/baseline/service.py` `_baseline_llm_caller` (lines 354-371), convert the system message into Anthropic content-blocks with `cache_control: {"type":"ephemeral"}`, and attach `cache_control: {"type":"ephemeral"}` to the last tool in the `tools` array.
2. Preserve `extra_body={"usage": {"include": True}}`.
3. Add a unit test in `backend/copilot/baseline/service_unit_test.py` asserting both markers are present on the `client.chat.completions.create` call.
4. Run `cd autogpt_platform/backend && poetry run format && poetry run pytest backend/copilot/baseline/ --no-header -q`.
5. Open a PR titled `perf(copilot/baseline): cache_control on system+tools for OpenRouter prompt caching`.

Verification (manual): run the local native stack, send "hello" twice to the /copilot chat, confirm `PlatformCostLog` shows first turn ~18.7K uncached input + ~$0.058 and second turn ~18K cacheRead + ~$0.006. Screenshot + paste cost numbers in the PR description.
```

---

## Task 2 — Delete `get_baseline_supplement()`

**Why:** [`autogpt_platform/backend/backend/copilot/prompting.py:443-454`](autogpt_platform/backend/backend/copilot/prompting.py#L443-L454) emits ~4,277 tokens of markdown describing tool behavior. Every one of those tools' schemas is ALREADY sent in the `tools:` array of the same baseline request. Anthropic-family models read the tools array directly; this supplement duplicates the information and pays for it every turn.

**Target files / lines:**
- `autogpt_platform/backend/backend/copilot/prompting.py:443-454` — function definition of `get_baseline_supplement`.
- `autogpt_platform/backend/backend/copilot/baseline/service.py` — remove the call site.
- Possibly `backend/copilot/tools/__init__.py` or `tools/models.py` — if any tool description currently lives in the supplement but should live on the tool schema itself, migrate it there first.

**Approach:**
1. Grep for `get_baseline_supplement` — find its single call site and remove it (and the import).
2. Read the full supplement content. For each instruction/rule it contains, check whether the corresponding tool's `description` field on the schema already covers it. If not, lift the missing guidance into the tool schema's `description` (keeping per-tool words short — Claude reads these).
3. Delete the function, its imports, and any tests that reference it.
4. **Behavior regression check:** run a set of copilot turns (`search_agents`, `run_agent`, `add_memory`, `forget_memory`, `search_memory`, `search_docs`, `read_workspace_file`) and confirm tool usage still works correctly. The supplement contained tool-selection guidance — if turns regress (picks wrong tool, skips a tool, etc.) restore the missing bits onto the relevant tool description.

**Measurement:** after removal, baseline "hello" input drops from ~18.7K to ~14.4K tokens. Cost $0.058 → $0.045 (~23% reduction).

**Risks:** Highest of the four tasks — behavioral regression in tool selection. Must verify with real turns covering each tool family before merging.

**Expected cost impact:** ~23% reduction per baseline turn. Compounds with Task 1.

### Agent prompt (Task 2)

```
Remove `get_baseline_supplement()` from the copilot baseline path. Work from `/Users/majdyz/Code/AutoGPT1`. Commit author: `majdyz <zamil.majdy@agpt.co>`. Use `--no-verify`. See `/Users/majdyz/Code/AutoGPT1/copilot-cost-followups.md` Task 2.

Scope:
1. Read `backend/copilot/prompting.py:443-454` (the full supplement body). For each rule/instruction, check if the relevant tool schema's `description` field in `backend/copilot/tools/**/*.py` already conveys it. If not, lift it into the schema description (keep descriptions concise).
2. Remove the call to `get_baseline_supplement()` from `backend/copilot/baseline/service.py` (grep for the call site).
3. Delete the function from `prompting.py` and any tests that directly reference it.
4. Run `cd autogpt_platform/backend && poetry run format && poetry run pytest backend/copilot/ --no-header -q`.
5. Manual regression sweep: start the local native stack, run one copilot turn that exercises each tool family (search_agents, run_agent, add_memory/forget_memory/search_memory, search_docs, read_workspace_file). Confirm tool selection + outputs match pre-change behavior. If a tool starts getting picked wrong or skipped, restore the missing guidance onto the relevant tool's schema description.
6. Open a PR titled `perf(copilot/baseline): drop get_baseline_supplement, schemas already cover tool guidance`.

Verification (manual): paste before/after `PlatformCostLog` costs for a "hello" turn into the PR description — expected drop from ~18.7K to ~14.4K input tokens.
```

---

## Task 3 — Shrink the Langfuse "CoPilot Prompt" v25

**Why:** The production Langfuse prompt compiles to ~5,016 tokens. Most of the bulk is the "Internal Reasoning Process" section (steps 1-10 with examples) and three full "Example Response Structure" blocks, which duplicate the same guidance with slightly different wording. This affects BOTH baseline and SDK turns — every turn pays for it.

**Target:**
- Langfuse project, prompt name "CoPilot Prompt", currently published at version 25.
- Prompt fetch: `backend/copilot/service.py:100-112` calls `_fetch_langfuse_prompt()`; on failure it falls back to the 329-token static `_CACHEABLE_SYSTEM_PROMPT`.

**Approach:**
1. Log in to Langfuse (production instance used by this deployment). Pull the v25 prompt source.
2. Identify duplicated sections. Typical targets:
   - Keep ONE "Internal Reasoning" description (not 10 enumerated steps with examples).
   - Keep ONE "Example Response" block, not three.
   - Collapse verbose "you should / you must" repetitions.
3. Preserve the core persona + guardrails — no safety-policy edits.
4. Publish as v26, do NOT delete v25 (rollback path).
5. Confirm the SDK path and baseline path both pick up v26 (via the Langfuse cache TTL at `langfuse_prompt_cache_ttl = 300s`, or restart backend to clear cache).

**Measurement:** before/after token count of the system prompt via `Anthropic.messages.count_tokens`. Target: 5K → ~2K.

**Risks:**
- Prompt surgery can regress model behavior in hard-to-predict ways. Spot-test a dozen varied turns before declaring done.
- This is a Langfuse edit, not a code change — no PR in this repo. Document the change in a PR title-only or in the team changelog.

**Expected cost impact:** ~3K fewer input tokens per turn on BOTH paths. Savings compound with Task 1 (baseline cache-read rate) and Task 4 (SDK cache-read rate).

### Agent prompt (Task 3)

```
Shrink the production Langfuse "CoPilot Prompt" (currently v25, ~5K tokens) to ~2K tokens while preserving persona + guardrails. This is a Langfuse edit, not a code PR. See `/Users/majdyz/Code/AutoGPT1/copilot-cost-followups.md` Task 3.

Scope:
1. Use the `mcp__langfuse__getPromptUnresolved` tool to fetch the current v25 body.
2. Identify duplicated sections — the "Internal Reasoning Process" with its 10 enumerated steps, the three "Example Response Structure" blocks, and any repeated "you must / you should" bullets.
3. Draft a v26 that keeps: persona statement, tool-use philosophy (concise), a single example response, and safety guardrails. Drop: step-by-step examples, alternate phrasings, duplicate blocks.
4. Publish v26 via `mcp__langfuse__createTextPrompt` or `mcp__langfuse__updatePromptLabels` — keep v25 unlabelled for rollback.
5. Verify token count with `poetry run python -c "from anthropic import Anthropic; print(Anthropic().messages.count_tokens(model='claude-sonnet-4-5', system=<body>, messages=[{'role':'user','content':'x'}]).input_tokens)"`. Target: ≤2,200.
6. Smoke-test the copilot with 5-10 varied turns (code question, general question, tool-using request, memory question, search request). Compare responses vs v25 for regression.

Post a summary (before/after token counts + spot-check transcripts) in the team changelog or Discord #copilot channel.
```

---

## Task 4 — Bump Claude Code CLI to ≥2.1.98 and remove the resume workaround

**Why:** The resume workaround at [`autogpt_platform/backend/backend/copilot/sdk/service.py:3040-3044`](autogpt_platform/backend/backend/copilot/sdk/service.py#L3040-L3044) disables `SystemPromptPreset` on every turn where `use_resume=True`. Resumed turns fall back to a raw-string system prompt with no `cache_control` markers → the SDK writes the full 33K prefix to cache every turn instead of reading a cross-user entry. CLI 2.1.98 introduced `--exclude-dynamic-system-prompt-sections` which fixes the crash the workaround exists for. CLI 2.1.116 is the latest stable.

**Target files / lines:**
- `autogpt_platform/backend/pyproject.toml:21` — `claude-agent-sdk = "0.1.58"` pin. The SDK bundles a specific CLI version.
- `autogpt_platform/backend/backend/copilot/config.py:217-230` — `claude_agent_cli_path` / `CHAT_CLAUDE_AGENT_CLI_PATH` override already exists; use it to point at a pinned CLI binary.
- `autogpt_platform/backend/backend/copilot/sdk/service.py:2881-2890` — drop `and not use_resume` from `_cross_user = config.claude_agent_cross_user_prompt_cache and not use_resume`.
- `autogpt_platform/backend/backend/copilot/sdk/service.py:3177-3186` — drop `and not ctx.use_resume` from `_cross_user_retry`.
- `autogpt_platform/backend/backend/copilot/sdk/cli_openrouter_compat_test.py` — existing CI gate for CLI bumps. Re-run against the new CLI.

**Approach:**
1. Install CLI 2.1.116 via `npm install -g @anthropic-ai/claude-code@2.1.116` or download the pinned binary into the deployment image.
2. Set `CHAT_CLAUDE_AGENT_CLI_PATH=/path/to/claude-2.1.116` in deployment env (or bump the SDK if a newer claude-agent-sdk release bundles 2.1.98+).
3. Run `poetry run pytest backend/copilot/sdk/cli_openrouter_compat_test.py -v`. If it passes → proceed. If regressions appear, fix them or fall back to 2.1.98 (smallest bump).
4. Remove the two `not use_resume` gates. Delete workaround comments referencing CLI 2.1.97.
5. Verify: run an SDK turn, then a follow-up in the same session. Turn 2's `PlatformCostLog` should show ~33K `cacheReadTokens` (not `cacheCreationTokens`). Expected cost drop: $0.21 → $0.01 per resumed turn.

**Risks:**
- CLI 2.1.116 may reintroduce OpenRouter-compat bugs the 2.1.97 pin was chosen for. `cli_openrouter_compat_test.py` is the gate — if it trips, fall back.
- Deployment image must ship the pinned CLI binary, not rely on the developer's globally installed `claude`.

**Expected cost impact:** ~95% drop on resumed SDK turns (every turn 2+ of every session).

### Agent prompt (Task 4)

```
Bump the Claude Code CLI to ≥2.1.98 (target 2.1.116) and remove the resume-workaround at sdk/service.py:3040-3044. Work from `/Users/majdyz/Code/AutoGPT1`. Commit author: `majdyz <zamil.majdy@agpt.co>`. Use `--no-verify`. See `/Users/majdyz/Code/AutoGPT1/copilot-cost-followups.md` Task 4.

Scope:
1. Install CLI 2.1.116 locally (`npm install -g @anthropic-ai/claude-code@2.1.116` or equivalent). Note the path.
2. Export `CHAT_CLAUDE_AGENT_CLI_PATH=<that path>` in your shell (or `.env`).
3. Run `cd autogpt_platform/backend && poetry run pytest backend/copilot/sdk/cli_openrouter_compat_test.py -v`. If any test fails, diagnose — the failures are likely the issues the workaround was protecting against. If fatal and unfixable, retry with CLI 2.1.98 (smallest bump that introduces the fix).
4. Remove `and not use_resume` from `_cross_user` at `backend/copilot/sdk/service.py:~2886`. Remove `and not ctx.use_resume` from `_cross_user_retry` at ~3182. Delete the workaround comment block referencing CLI 2.1.97.
5. Update `autogpt_platform/backend/pyproject.toml:21` comment if it still says "bundled CLI 2.1.97".
6. Update deployment: make sure the Docker image (or wherever the backend runs in prod) ships CLI 2.1.116 and sets `CHAT_CLAUDE_AGENT_CLI_PATH` appropriately.
7. Run `poetry run format && poetry run pytest backend/copilot/sdk/ --no-header -q`.
8. Manual verification: start the local native stack, run an SDK-mode copilot turn, then a follow-up turn in the same session. In the `PlatformCostLog` table, confirm turn 2 shows `cacheReadTokens > 0` (not just `cacheCreationTokens`). Expected per-turn cost: $0.21 → $0.01.
9. Open a PR titled `perf(copilot/sdk): bump CLI to 2.1.116, unlock cross-user cache on resumed turns`.
```

---

## Recommended merge order

1. **Task 1** first — smallest, direct cost reduction, easy to verify.
2. **Task 4** — biggest win per turn, but requires deployment plumbing + compat test re-validation.
3. **Task 3** — prompt surgery; safe to do anytime, doesn't block the code changes.
4. **Task 2** last — highest regression risk (needs manual sweeps across all tool families).

## Appendix: useful commands

```bash
# Token count a system prompt
cd /Users/majdyz/Code/AutoGPT1/autogpt_platform/backend && poetry run python -c "
from anthropic import Anthropic
body = open('/tmp/system.txt').read()
print(Anthropic().messages.count_tokens(model='claude-sonnet-4-5', system=body, messages=[{'role':'user','content':'x'}]).input_tokens)
"

# Inspect a Langfuse trace
cd /Users/majdyz/Code/AutoGPT1/autogpt_platform/backend && poetry run python -c "
from langfuse import Langfuse
lf = Langfuse()
trace = lf.fetch_trace('<trace_id>')
print(trace.data.model_dump_json(indent=2))
"

# Check cost of last N turns for a user
docker exec supabase-db psql -U postgres -d postgres -c "SELECT \"createdAt\", \"blockName\", \"costMicrodollars\", \"inputTokens\", \"cacheReadTokens\", \"cacheCreationTokens\" FROM platform.\"PlatformCostLog\" WHERE \"userId\" = '<uid>' ORDER BY \"createdAt\" DESC LIMIT 10;"

# Inspect Redis rate-limit counter
docker exec autogpt_platform-redis-1 redis-cli MGET "copilot:cost:daily:<uid>:$(date -u +%Y-%m-%d)" "copilot:cost:weekly:<uid>:$(date -u +%Y-W%V)"
```
