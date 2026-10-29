---
name: perf-compare
description: "Shared measurement recipe for AutoGPT performance work. Compares a baseline and a candidate in Sentry (two releases, two environments such as preview pr-<n> deploys, or two time windows) per transaction: count, p50/p95/p99, failure rate, and a trace-clustered bootstrap verdict of faster / slower / no detectable change / not enough data, with exit codes agents can branch on. Works for span duration, web vitals (measurements.lcp) or any numeric span field. TRIGGER when an agent needs to show a change made the backend or frontend faster or slower, compare a preview against a control or dev, check a release after merge on dev or prod, or check for regressions."
user-invocable: true
argument-hint: "[what to compare, e.g. 'builder release OLD vs NEW on prod' or 'pr-123 vs control pr-124 on /api/library/agents']"
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# perf-compare: one way to measure performance

Every agent that tunes speed or stability measures with this recipe, so results are comparable across agents and noise is not reported as a win. Sentry is the store: org `significant-gravitas`, API `https://us.sentry.io`, tracing at 100%. `scripts/perf_compare.py` (Python 3.8+, standard library only) runs the queries and the statistics.

```bash
S=$(git rev-parse --show-toplevel)/.claude/skills/perf-compare/scripts/perf_compare.py
python3 "$S" --help
```

## When to use it

- Before claiming a change made something faster or slower, on a preview or after merge.
- After merge, to check a win holds on dev and then prod (release against release).
- To check a change did not regress other transactions (run without `--transaction`).
- For any numeric field: `span.duration` (default), web vitals on builder pageloads (`measurements.lcp`), LLM token counts (`gen_ai.usage.input_tokens`), or custom attributes (`tags[name,number]`).

It is not for debugging one slow request (open the trace) or for error triage.

## Token

The script reads `SENTRY_AUTH_TOKEN` from the environment and nowhere else. It needs a read-only token with exactly `org:read`, `project:read` and `event:read`: either a user auth token (Settings > Account > API > Auth Tokens) or an internal integration with Read on Organization, Project and Issue & Event. Organization auth tokens (`sntrys_...`) carry only `org:ci` and get HTTP 403 on these endpoints. A missing token exits with code 5 and these instructions. `--print-queries` needs no token.

## Where things are in Sentry

| Project | What it holds | Environments | `release` |
|---|---|---|---|
| `autogpt-server` | Backend: FastAPI routes (`/api/library/agents`, `/api/chat/sessions/{session_id}/stream`) and internal RPC services (`/update_node_execution_status`) | `app:prod-behave:cloud`, `app:dev-behave:cloud` | git sha once the backend sets it; check first |
| `builder` | Frontend (Next.js on Vercel): `http.server` routes (`GET /api/proxy/[...path]`), `middleware GET`, `serverAction/*`, browser `pageload` and `navigation` with web vitals | `app:prod-behave:cloud` is prod; `app:preview-behave:cloud` is dev **and** every Vercel preview | git sha |
| `autogpt-server-previews`, `builder-previews` | `!deploy` previews | `pr-<n>` | git sha |

Check a selector has data before you measure:

```text
search_events  projectSlug=<project>  dataset=spans  query="is_transaction:true"
               fields=["environment","release","count()"]  sort="-count()"  period="7d"
```

If `release` comes back null for a project, select by time window instead of release. If `autogpt-server-previews` has no `pr-<n>` events, that preview's backend is reporting into `autogpt-server` as `app:dev-behave:cloud`, mixed with dev traffic, and cannot be measured separately.

## Choosing baseline and candidate

Both sides take `KEY=VALUE` pairs: `project=`, `env=`, `release=`, `query=` (extra Sentry filter), and time as `window=START/END` (repeatable), `start=`/`end=`, or `period=24h`. Shared defaults come from `--project`, `--env`, `--query` and `--period`/`--start`/`--end`/`--window`. A side's own `project=`, `env=` or `query=` wins over the shared value (`env=` with nothing after it clears it). Any time key on a side (`window=`, `start=`/`end=` or `period=`) replaces all of the shared time flags for that side. Times are UTC ISO 8601; without a time anywhere, both sides use the last 7 days.

**1. After merge: release against release (dev or prod).** Same project and environment, previous release against new release, in windows of equal length that cover the same hours of the day: the 24 hours before the deploy against the 24 hours after, or 7 days against 7 days for quiet routes. Always filter by release when there is one. Old releases keep sending events for days (open tabs, Vercel skew protection), so a time window alone mixes releases.

```bash
python3 "$S" --project builder --env app:prod-behave:cloud \
  --baseline release=OLD_SHA window=2026-10-10T14:00:00Z/2026-10-11T14:00:00Z \
  --candidate release=NEW_SHA window=2026-10-11T14:00:00Z/2026-10-12T14:00:00Z
```

**2. On a preview: candidate preview against a control preview.** Deploy the change with `!deploy`, and deploy a control (the dev commit the change branches from, with no changes) the same way. Each gets its own namespace and an empty branch database, so the two are alike. Drive the same load at both, either together in one window or interleaved A B B A (control, candidate, candidate, control; about 10 minutes each) so drift cancels:

```bash
python3 "$S" --project autogpt-server-previews --transaction '/api/store/agents' \
  --baseline env=pr-CONTROL window=T0/T1 window=T3/T4 \
  --candidate env=pr-CANDIDATE window=T1/T2 window=T2/T3
```

Measuring the candidate preview against `app:dev-behave:cloud` at dev HEAD in the same window is cheaper but weaker: dev has a full database, real traffic and different pods. Use it as a rough read, not as proof.

**3. No release tag: before against after.** Equal windows either side of the deploy. Other deploys inside those windows also change things, so say so in your report.

**4. A/A check before trusting a win.** Compare a selector with itself over two windows, for example the same release yesterday and today. It should say `no detectable change`. If it does not, your windows differ too much (hour of day, weekday, traffic mix), and so will your real comparison.

## How much data

- The script gives no p50 or p95 verdict unless each side has `--min-count` samples (default 200) **from** `--min-traces` distinct traces (default 30). At 200, p95 rests on 10 values above it. p99 is shown but never judged. The verdict cell says which minimum was missed. The failure-rate test works from Sentry's event counts instead, and needs `--min-count` events per side.
- It samples up to `--sample-cap` values per side per transaction (default 1000, at most 20000), paging 100 at a time. A cap below `--min-count` is refused.
- For a claim about one transaction, name it with `--transaction` and raise `--sample-cap` to 2000–5000. Naming it also shrinks the multiple-comparison correction (see below).
- Correlated spans count once: the bootstrap resamples whole traces, and the trace minimum stops a few busy traces (a page polling an endpoint) from passing as a big sample. The table shows samples and distinct traces. The effective sample is closer to the trace count than the span count. In A/A tests on clustered data, resampling spans one at a time called 38–44% of identical pairs a change, against 2–5% when resampling by trace.
- Drive at least 1000 requests per side per transaction you care about. Wait a few minutes after load stops before measuring, so Sentry has the last events.

## Driving load on a preview

A preview's backend is `https://pr-<N>-server.agpt.co` (REST under `/api`, health at `/health`), and its frontend is `https://autogpt-pr-<N>.vercel.app`.

The k6 tests are in `autogpt_platform/backend/load-tests/`; its `README.md` covers every test and setting. Run them on your own machine: `./k6.sh` runs k6 in Docker (`grafana/k6`), so nothing else needs installing, and a local run costs nothing beyond the load on the target. Do not run them on Grafana Cloud k6. It bills every virtual-user hour, so `./k6.sh` refuses cloud runs unless they are explicitly opted into, and the README reserves them for rare runs the team has agreed to. `K6_ENVIRONMENT=PREVIEW` with `PR_NUMBER=<N>` targets a preview:

```bash
cd autogpt_platform/backend/load-tests
export K6_ENVIRONMENT=PREVIEW PR_NUMBER=<N>
VUS=5 DURATION=10m RAMP_UP=30s RAMP_DOWN=30s ./k6.sh run tests/api/core-api-test.js
```

- **Authentication needs no setup on a preview.** The authenticated tests (`tests/api/*`, `tests/basic/*`, `tests/marketplace/library-access-test.js`, `tests/comprehensive/platform-journey-test.js`) sign up throwaway `loadtest-*@previews.agpt.co` accounts in `setup()`, once per run, and use the Better Auth tokens the preview's frontend issues for them. `LOAD_TEST_ACCOUNT_COUNT` sets how many (default 5). `tests/marketplace/public-access-test.js` needs no account at all.
- **To run as specific accounts**, such as the seeded `preview-*@previews.agpt.co` personas, set `LOAD_TEST_ACCOUNTS` to a JSON array of `{"email", "password"}`. The tests sign each one in once per run. Dev has no throwaway sign-up, so it always needs `LOAD_TEST_ACCOUNTS`.
- **Note the run's UTC window** (`date -u +%FT%TZ` before and after) for the baseline and candidate windows above, and drive the same test, VUs and duration at both sides.
- **Previews start with an empty database.** Store and library lists return nothing, so costs that grow with data (N+1 queries, scans) do not show. Seed data through the API first, or confirm on dev and prod after merge.
- None of the k6 tests open copilot chats. To measure copilot turns, run chats against the preview yourself, or measure on dev and prod traffic.
- The repository's evals (`backend/copilot/eval`, `backend/copilot/capabilities/eval`, `poetry run expert-style-eval`, `poetry run brain-dump-eval`) are offline quality and cost harnesses. They send no traffic to a deploy.
- After merge, real traffic on dev and prod is the load.

## Reading the verdict

The markdown output (default) or `--json` reports two things for each transaction.

- **Sentry aggregates over every matching event:** count, p50, p95, p99 and failure rate. These are exact for the windows.
- **A verdict for p50 and p95:**
  - The interval is a bootstrap CI of candidate minus baseline from the raw samples, resampling whole traces.
  - It is Bonferroni-adjusted so that every interval in the run holds together at `--confidence` (95%). Two transactions means four intervals at 98.75% each. The plain 95% interval is in the JSON as `ci`.
  - Resamples are raised automatically until each tail of the adjusted interval holds 20 draws, up to 50000. For example, 4000 becomes 8000 for five transactions. At the 50000 cap a run still goes ahead with as few as 10 draws per tail. A run that would leave fewer than 10, because too many transactions are compared, is refused before it starts (exit 2). The JSON reports the number used as `settings.resamples`.
  - `faster` or `slower` needs the adjusted interval to exclude zero, and **both** the sample change and Sentry's all-events change to be at least `--min-effect` (5%) in that direction. Anything else is `no detectable change`. Below `--min-count` samples or `--min-traces` traces it is `not enough data`.
  - The transaction takes its worst percentile: `slower` if either is slower, else `faster` if either is faster.
- **Failure rate:** Fisher's exact test on the failure counts behind Sentry's `failure_rate()`. It is exact because a normal approximation is wrong with a handful of failures: 0 of 1000 against 4 of 1000 passes a z-test, but Fisher gives p = 0.12.
  - The significance level is Bonferroni-adjusted over the K transactions that have `--min-count` events on both sides. That is one test per transaction, not two.
  - `higher` needs significance and a rise of at least `--min-failure-delta` (0.1 percentage points). It counts as a regression even when the transaction got faster, because requests that fail fast look like a speedup.
  - The test treats every event as independent, and `--min-traces` does not apply to it. When failures come from a few traces (one broken page retrying), it overstates significance. Check `sample_traces` and the failing traces in Sentry before acting on a failure-rate result alone.

`no detectable change` means "not distinguishable from noise with this much data". It does not mean "no change". Check the interval widths and trace counts before concluding anything.

**A win counts** only when all of these hold:

- you named the transaction with `--transaction` before measuring;
- the verdict is `faster`;
- the failure rate is not `higher`;
- it holds again on a second, independent window or run.

A run over the top transactions is for spotting regressions and leads, not for claiming wins.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | faster: a compared transaction got faster and nothing regressed |
| 1 | slower or more failures: at least one compared transaction regressed |
| 2 | usage error: bad arguments (including a `--region-url` that is not https, `--timeout` ≤ 0, or `--sample-cap` below `--min-count`), unknown project, or Sentry rejected the query (HTTP 400) |
| 3 | no detectable change |
| 4 | not enough data: no compared transaction met `--min-count` and `--min-traces` on both sides |
| 5 | auth: `SENTRY_AUTH_TOKEN` missing, or Sentry rejected it (401/403) |
| 6 | Sentry API or network error after retries, or any unexpected error; never read as a regression |

With `--json`, the fields to branch on are:

- `overall.verdict`, `overall.exit_code` and `overall.reasons`;
- per transaction in `transactions[]`:
  - `verdict`, `note` (why there is not enough data), `regression`;
  - `p50`/`p95`: `baseline`, `candidate`, `diff`, `rel`, `sentry_rel`, `ci`, `ci_adjusted`, `verdict`;
  - `failure_rate`: `baseline`, `candidate`, `diff`, `p_value`, `verdict`;
  - `baseline`/`candidate`: `count`, `p50`, `p95`, `p99`, `failure_rate`, `sample_n`, `sample_traces`;
- `settings`, holding the values actually used: `sample_cap`, `resamples`, `comparisons`, `failure_comparisons`, `adjusted_confidence`, `min_count`, `min_traces`, `min_effect`, `seed`.

Errors in `--json` mode print `{"error": {"type", "message"}, "exit_code"}`, where `type` is `usage`, `auth`, `api` or `internal`.

## The exact queries

All requests are `GET https://us.sentry.io/api/0/organizations/significant-gravitas/events/` with `dataset=spans` and `project=<numeric id>`. The id comes from `GET /api/0/projects/significant-gravitas/<slug>/`. `start` and `end` (UTC) span the side's windows; the query's timestamp filter pins each window exactly. `F` is `--field`.

1. **Aggregate per transaction**, one request per side:
   - `field=transaction`, `count()`, `p50(F)`, `p95(F)`, `p99(F)`, `failure_rate()`
   - `sort=-count()`, `per_page=100`
2. **Raw sample**, per side per compared transaction:
   - `field=F`, `trace`, `id`
   - `sort=id`, `per_page=100`, paged with the `cursor` from the `Link` header (`rel="next"; results="true"`) until `--sample-cap` rows.
   - Span ids are random, so the first N rows by `id` are a uniform random sample of the windows. Sorting by `-timestamp` instead returns the latest burst, often a few traces.

Both use this query string:

```text
is_transaction:true environment:"ENV" release:"SHA" [has:F] [(EXTRA QUERY)]
  [transaction:"T" | (transaction:"A" OR transaction:"B")]
  timestamp:>=2026-10-11T14:00:00Z timestamp:<2026-10-12T14:00:00Z
```

Several windows become `(timestamp:>=S1 timestamp:<E1 OR timestamp:>=S2 timestamp:<E2)`. `has:F` appears for any field other than `span.duration`.

`python3 "$S" ... --print-queries` prints the exact REST URL and MCP arguments for an invocation, without a token.

**With only the Sentry MCP**, use `search_events` with `dataset: "spans"`, or `search_traces` without `dataset`:

- Pass `projectSlug`, the same `query`, `fields` and `sort`.
- Pass a `period` long enough to cover the windows, such as `"50h"`; the timestamp filter does the exact cut. The argument is `period`, not `statsPeriod`, and an unknown key silently falls back to 14 days.
- Pass `limit` ≤ 100.
- Confirm the "Executed Search" block in the reply shows your query unchanged.
- The MCP cannot page. For more than 100 samples, split each window into sub-windows, get `count()` for each, and take the first rows by `id` from each in proportion to its count.

## Field names checked against live data

Checked through the Sentry MCP on `builder` and `autogpt-server`.

- **Filters:**
  - `is_transaction:true` and `transaction:"…"`.
  - `environment:"…"`: quote it, the values contain colons.
  - `release:"<sha>"`.
  - `span.op:` with `pageload`, `navigation`, `http.server`, `http.server.middleware` or `function.server_action`.
  - `span.status:[…]` and `!span.status:[…]`.
  - `has:measurements.lcp` and `has:tags[gen_ai.usage.input_tokens,number]`.
  - `timestamp:>=…Z timestamp:<…Z` and parenthesised `OR` groups.
- **Aggregates:**
  - `count()`.
  - `p50()`, `p75()`, `p95()` and `p99()`, of `span.duration` (ms), `measurements.lcp`, `measurements.fcp`, `measurements.ttfb`, `measurements.cls`, `gen_ai.usage.input_tokens` or `tags[gen_ai.usage.input_tokens,number]`.
  - `failure_rate()`: the share of spans whose `span.status` is not `ok`, `cancelled` or `unknown`. It reconciles exactly with `count()` and `!span.status:[ok,cancelled,unknown]`.
- **Sample fields:** `span.duration`, `trace`, `id`, `timestamp`, `release`. Sorting works by `id`, `-id` and `-timestamp`.
- **`span.status` values seen:** `ok`, `internal_error`, `resource_exhausted`, `cancelled`, `deadline_exceeded`, `invalid_argument`, `not_found`, `already_exists`.
- **Rejected:** wildcards on `id` such as `id:0*`; span ids must be full 16-character hex.

## Pitfalls

- **Environment names collide.** Builder's dev deploy and every Vercel preview share `app:preview-behave:cloud`. Tell them apart with `release`.
- **Proxy routes measure the backend.** `GET /api/proxy/[...path]` mostly measures the backend behind it, not the frontend release, and a few polling pages contribute most of its spans.
- **Middleware durations are quantized** to about 1 ms. A one-step move in p95 reads as +100%; look at the absolute change.
- **Long-lived transactions measure session length.** `/ws` and `/api/chat/sessions/{session_id}/stream` last minutes to an hour, not request latency. Compare per-turn or per-node transactions instead.
- **Weekday and hour of day move latency.** Saturday-to-Sunday against Sunday-to-Monday is not a fair pair for a release comparison. Use full days, the same hours, and the A/A check.
- **Do not rerun until it says faster.** Repeating runs, or trying many transactions and reporting the best, turns noise into wins. Decide the transaction, windows and settings first, then run once and replicate once.

## Tests

```bash
python3 -m unittest discover -s .claude/skills/perf-compare/scripts
```

The tests run offline in a few seconds, against the synthetic Sentry responses in `scripts/fixtures/`, served by a fake transport or by a local `http.server`. They cover:

- query building, time parsing and option validation;
- cursor pagination, retries, and real-socket errors;
- the percentile, bootstrap and Fisher maths with fixed seeds;
- seeded A/A calibration on independent and trace-clustered data;
- the verdict rules, every exit code, and the missing-token error.
