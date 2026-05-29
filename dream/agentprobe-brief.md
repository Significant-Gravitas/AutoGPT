# AgentProbe — Memory Tests + Ranking Tooling Brief

You are a persistent sub-agent commissioned to extend [AgentProbe](https://github.com/Significant-Gravitas/AgentProbe) with two things AutoGPT's dream-system memory work needs and AgentProbe doesn't have yet:

1. **Retrieval-ranking scorer** — precision@k, recall@k, MRR, NDCG@k. AgentProbe today scores everything via LLM-judge over a transcript; there is no scorer that grades "did the right thing come back in the right order from a list."
2. **Memory-system test scenarios** that exercise dream-pass behaviors (typed-edge filtering, retract vs. soft-delete, scoped cascading expiry, stale-fact deprecation, demotion-correctness) — some using the new ranking scorer, some using the existing LLM-judge.

Read the **primer** at `/Users/ntindle/code/agpt/AutoGPT/branch8/dream/agentprobe-primer.md` *before you start* — it has the repo orientation, file anchors, conventions, and the dream-system context that justifies what you're building. Then read `/Users/ntindle/code/agpt/AutoGPT/branch8/dream/dreaming-agentprobe-evals.md` for the existing survey of AgentProbe (what's there, what's missing, what we want to preserve).

## Mission

Ship a single self-contained PR to `Significant-Gravitas/AgentProbe` that:

1. Adds a **`kind: ranking`** scorer discriminator (or whatever idiomatic AgentProbe naming is for "non-judge scorer") that computes precision@k, recall@k, MRR, NDCG@k from a list of expected items and a list of returned items.
2. Adds YAML schema support so scenarios can declare a `ranking:` block alongside or instead of `rubric:`. Example shape, but defer to what's idiomatic for AgentProbe:

   ```yaml
   - id: mem-retrieval-warm-context
     scorer:
       kind: ranking
       golden:
         - "Sarah's email"          # text OR uuid; canonical comparison
         - "Atlas project status"
       k: 5
       weight:
         precision_at_k: 1.0
         recall_at_k: 1.0
         mrr: 1.0
         ndcg_at_k: 1.0
   ```
3. Adds **at least 5 memory test scenarios** that use the new ranking scorer. Prioritize coverage of:
   - **Forget-on-request** (already exists at `data/multi-session-memory.yaml:1710` as a judge-scored scenario — add a ranking-scored sibling that asserts the forgotten item is **not** in the top-k)
   - **Warm-context relevance** — given session history, a query, and a golden expected set, score the retrieved memories
   - **Stale-fact demotion** — superseded facts should not appear in the top-k
   - **Scope filtering** — `project:foo` query should not surface `project:bar` memories
   - **Cascading expiry** — invalidating an entity should remove its directly-attached facts from the top-k but **not** tangentially-related ones
4. Persists the new metrics in the database alongside the existing `judge_dimension_scores` / `human_dimension_scores`. Add a new `retrieval_scores` table (or reuse the existing dimension-score table if that's idiomatic — *judge the AgentProbe codebase's preference*).
5. Surfaces the metrics in any text/JSON reports. Dashboard work is **optional** — only do it if the existing dashboard code is structured such that a one-shot addition is trivial. Otherwise note it as a follow-up.

## Hard requirements

- **Branch:** new branch off fresh `origin/main` (e.g. `feat/ranking-metrics`).
- **Pull cleanly first:** `git fetch --all && git checkout main && git pull --no-rebase`. **Never rebase.** **Never force-push.** **Never `git push --force`.** If the working tree is dirty before pull, stop and report — do NOT stash, reset, or discard work.
- **Use `--no-verify` on commits** — pre-commit hooks are broken locally on this machine (Python 3.14 pyexpat symbol mismatch).
- **Conventional commits** with scope (e.g. `feat(scorer): add ranking metrics`).
- **Each commit independently green** (`bun test`). If a commit breaks tests, fix in the same commit (amend before next commit) — but **never amend a commit you have already pushed**.
- **Comprehensive PR description** — what / why / how, risk note, out-of-scope list, checklist. Use the existing PR template if there is one. Pass body via `--body-file <tempfile>` to `gh pr create` (NOT `--body` — backticks get eaten by shell).
- **Tests for all new code paths.** Vitest/bun-test style; place tests next to source (`*.test.ts`) per AgentProbe convention.
- **Never commit secrets** — if you find any `.env` files with real keys, treat as background config to read but never to commit. Use `.env.example`-style placeholders if a new env var is needed.
- **No emoji** in code, commits, PR descriptions, or anywhere unless AgentProbe's existing conventions explicitly use them.

## Success criteria

- [ ] Ranking scorer implemented behind a clean discriminator; covers precision@k / recall@k / MRR / NDCG@k.
- [ ] YAML schema parses + validates the new shape; existing scenarios untouched.
- [ ] ≥5 new memory scenarios using the ranking scorer (or extensions of existing ones).
- [ ] Tests in place for the metrics math (well-known small examples — e.g. NDCG of `[1, 0, 1]` with binary relevance and `log2` discount).
- [ ] Persistence: scores stored and retrievable.
- [ ] PR open against `Significant-Gravitas/AgentProbe` `main` with a thorough description.
- [ ] No regression — `bun test` green; existing scenarios still parse and score.

## Out of scope

- License/IP work for AgentProbe ↔ AutoGPT (separately tracked).
- Anything in `AutoGPT` itself.
- Frontend dashboard polish beyond a one-shot addition.
- Adding the remaining 24 unused multi-session-memory scenarios — pick the most useful additions and stop.
- A whole new adapter type — work within existing adapter abstractions (HTTP, WS, OpenClaw harness).

## When done

Post a final message summarizing: PR URL, commits list, what's covered, what was deferred, and any concerns or judgment calls that need human review.

If you hit a blocker (license question, AgentProbe code style ambiguity, missing dependency), pause and write a clear question in your output describing the choice and the options. Don't guess on judgment calls that affect API shape.
