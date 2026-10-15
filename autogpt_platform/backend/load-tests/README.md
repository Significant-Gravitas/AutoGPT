# AutoGPT Platform Load Tests

[k6](https://grafana.com/docs/k6/latest/) tests that drive the platform's user-facing workload at a deployed target, a `!deploy` preview as a rule, and measure what users feel: time to first token in copilot, how long a graph run takes to start and finish, how fast the store, library and account pages answer. The backend reports every request to Sentry, and the [perf-compare skill](../../../.claude/skills/perf-compare/SKILL.md) decides whether two builds differ.

Run them on your own machine with Docker. Don't run them on Grafana Cloud k6: it bills every virtual-user hour (see [What it costs](#what-it-costs)).

## Quick start

Docker is the only requirement. `./k6.sh` runs k6 in a container with this directory mounted and takes the same arguments as `k6`. Its first run builds the image, the pinned `grafana/k6` plus the [xk6-sse](https://github.com/phymbert/xk6-sse) extension that reads copilot streams as they arrive, in about two minutes.

1. Comment `!deploy` on the pull request. A few minutes later its backend answers at `https://pr-<N>-server.agpt.co/health` and its frontend is `https://autogpt-pr-<N>.vercel.app`.
2. Run the smoke profile, a minute of every journey at a low rate, to see that the target and the seeding work:

   ```bash
   cd autogpt_platform/backend/load-tests
   export K6_ENVIRONMENT=PREVIEW PR_NUMBER=<N>
   LOAD_PROFILE=smoke ./k6.sh run tests/workload.js
   ```

   `setup()` runs before the load: signing up and subscribing the accounts, seeding them and waiting for the seed runs takes about nine minutes with the default eight accounts.

3. Run the baseline profile, the comparison run: a one-minute warm-up, then ten minutes at the rates below.

   ```bash
   LOAD_PROFILE=baseline ./k6.sh run tests/workload.js
   ```

Each run prints a report and writes `results/<RUN_ID>.json` (see [The summary](#the-summary)); the log says why any journey skipped (`K6_LOG_OUTPUT=file=results/<RUN_ID>.log` keeps it). The run signs up its own accounts and seeds its own data. Previews paywall new accounts, so for graph runs and chat sessions export `STRIPE_TEST_SECRET_KEY`, a test-mode key of the Stripe account previews use: the run then subscribes its accounts in Stripe test mode and cancels the subscriptions at the end (see [Paywalled targets](#paywalled-targets)). Copilot turns also need the preview's dummy engine (see [Copilot and model spend](#copilot-and-model-spend)). Comment `!undeploy` when you are done with the preview.

## The workload

`tests/workload.js` runs six journeys at once, each as an open-model k6 scenario (`ramping-arrival-rate`): iterations start at a fixed rate whatever the target's latency, so a slower build cannot quietly send fewer requests. Queueing shows up as latency, and, if every virtual user is busy, as `dropped_iterations`, which fails the run.

The rates are iterations per minute at the baseline profile, weighted by each journey's share of user-facing requests in production (Sentry, `autogpt-server`, `app:prod-behave:cloud`, seven days). Rare but slow journeys get more than their share so a ten-minute run reaches a usable sample.

| Journey          | Rate | What one iteration does                                                                                                                                                                                                                                | Why this weight                                                                                                                                  |
| ---------------- | ---: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `copilot_reads`  |   60 | Opens the copilot page: `GET /chat/sessions`, `/chat/usage`, `/chat/connections`, `/experts/identities`, then one session's detail                                                                                                                     | Chat session list, usage and session detail are the largest family of user-facing requests (about 35%)                                           |
| `copilot_turn`   |   15 | A conversation turn: `POST /chat/sessions` (or reuses one of the account's seeded sessions), `POST /chat/sessions/{id}/stream`, reads the Server-Sent Events to `finish`, then `GET /chat/sessions/{id}`                                               | Rare by count in production (about a hundred streams a day) but it is the product and its cost; the rate is a sample-size floor                  |
| `account_reads`  |   30 | What every page load asks: `GET /credits`, `/credits/subscription`, `/credits/trial`, `/onboarding`, `/onboarding/completed`, `/integrations/credentials`, `/integrations/providers`, `/integrations/providers/system`, `/orgs`, `/auth/user/timezone` | The second-largest family (about 33%), and `/credits/subscription` is the slowest common read in production                                      |
| `store_browse`   |   30 | Browses the marketplace without signing in: `GET /store/skills`, a skill's detail, `GET /store/agents`, an agent's detail, a search, the creators list; then the account's `GET /store/profile`                                                        | Skills list and detail, store agents and search are about 25% of requests                                                                        |
| `library_browse` |   24 | `GET /library/agents`, a library search, then one agent's detail                                                                                                                                                                                       | Under 1% of requests, but p50 of 11 s for the list and 39 s for the detail in production                                                         |
| `graph_run`      |   10 | Opens one of the account's seeded graphs (`GET /graphs/{id}`), opens the WebSocket and subscribes to it, starts a run of it, waits for the run's events to say it completed, confirms with `GET /graphs/{id}/executions/{exec}`, and lists executions  | 130 to 210 runs a day in production, started by webhooks, schedules and copilot; this is the executor path, with the WebSocket users watch it on |

The graphs are built from deterministic blocks that do real work without a model or credentials (lib/graphs.js), so a run costs nothing and finishes in a known number of steps.

### Profiles

| `LOAD_PROFILE` | Rates          | Warm-up | Steady | Thresholds  | Use                                                   |
| -------------- | -------------- | ------- | ------ | ----------- | ----------------------------------------------------- |
| `smoke`        | 0.1 x baseline | none    | 1 min  | 3 x relaxed | Does the target work, does the seeding work           |
| `baseline`     | the table      | 1 min   | 10 min | as declared | The comparison run; about 16 requests a second in all |
| `stress`       | 3 x baseline   | 1 min   | 10 min | 2 x relaxed | What breaks first                                     |

The warm-up ramps each rate up from zero. Requests made during it carry `phase:warmup`; every threshold and every per-endpoint and user-facing number in the summary covers `phase:steady` only. The per-journey iteration times (`by_journey`) span the whole run, since k6 times an iteration without the request tags.

Settings: `SCENARIOS=copilot_turn,store_browse` runs a subset; `RATE_SCALE=0.5` multiplies every rate; `WARMUP` and `DURATION` replace the phase lengths; `THRESHOLD_SCALE` relaxes the thresholds; `RUN_SEED` fixes which items each iteration picks (default fixed, so two runs send the same sequence); `LOAD_TEST_ACCOUNT_COUNT` is the number of accounts (default 8).

### What is measured

- Every request carries `endpoint` (such as `store.skill_detail`) and `phase` tags, and one check named after the endpoint. Thresholds in `configs/thresholds.js` set a p95 per endpoint, above production's p95 for the route, and a failure rate under 2%. The graph run's own numbers and its execute route are sized on a preview instead, whose executor is smaller than production's: a run of a seeded graph takes about 30 s there.
- Custom Trends for what users feel: `copilot_first_chunk_ms` (the first frame), `copilot_ttft_ms` (the first text), `copilot_turn_ms` (the `finish` frame), `copilot_chunks_per_turn`; `graph_run_start_ms` (execute to the first running event) and `graph_run_ms` (execute to completed); `ws_connect_ms` and `ws_event_latency_ms` (the event's `ended_at` to its arrival). Counters tell completed, queued and failed copilot turns and graph runs apart.
- `copilot.stream`'s `http_req_duration` is the whole stream, so it has no latency threshold; the Trends above are its numbers.

### The summary

`handleSummary` writes `results/<RUN_ID>.json` with:

- `run_id`, `profile`, `seed`, the scenarios and the rate each ran at;
- `target`: name, URLs and the Sentry project and environment to query (`autogpt-server-previews`, `pr-<N>`); `target.build` is the backend's `/health` answer, which does not name the commit, so select the preview's traces by environment and window (its Sentry `release` is the git sha);
- `window`: `started_at`, `load_started_at` (after setup) and `ended_at`, UTC, for perf-compare's `window=`;
- per endpoint and per custom metric: count, p50, p95, p99, failure rate, steady phase only;
- `by_journey`: each journey's iteration time over the whole run and its dropped iterations;
- the seeding counts, including the subscriptions created and cancelled and how many store agents and skills the target had;
- `coverage`: per journey, `covered`, `partial` or `skipped`, with the reason when it is not covered (the paywall, no dummy engine, an empty marketplace, no sessions); the text report prints the same under `NOT COVERED IN FULL`, because a list read against an empty marketplace is fast for the wrong reason;
- every threshold's result.

`RUN_ID` comes from `./k6.sh` (a timestamp) unless you set it. The summary and the run's accounts are named after it.

## Comparing two builds

The verdict comes from the backend's traces in Sentry through perf-compare, not from k6's client-side numbers: perf-compare resamples by trace, corrects for multiple comparisons and refuses to judge on too little data. The procedure:

1. Deploy the candidate with `!deploy`, and deploy a control the same way: a pull request at the dev commit the candidate branches from, with no changes. Each preview gets its own namespace and database, so they are alike.
2. Run the baseline profile at both, interleaved, with the same seed. `scripts/ab.sh` does this (A B B A by default, about 45 minutes) and prints the perf-compare command for the windows it produced:

   ```bash
   scripts/ab.sh <control PR> <candidate PR>
   ```

   By hand, the equivalent is four `LOAD_PROFILE=baseline ./k6.sh run tests/workload.js` runs alternating `PR_NUMBER`, taking each run's window from its summary.

3. Wait a few minutes for Sentry to ingest the last events, then name the transaction you care about and compare:

   ```bash
   S=$(git rev-parse --show-toplevel)/.claude/skills/perf-compare/scripts/perf_compare.py
   python3 "$S" --project autogpt-server-previews --transaction '/api/store/skills/{slug}' \
     --baseline env=pr-<CONTROL> window=<START1>/<END1> window=<START4>/<END4> \
     --candidate env=pr-<CANDIDATE> window=<START2>/<END2> window=<START3>/<END3>
   ```

   For copilot, compare the executor's `copilot.turn` transaction and its span data: `--transaction copilot.turn --field 'tags[copilot.turn.ttft_ms,number]'` (also `copilot.turn.queue_wait_ms`, `copilot.turn.duration_ms`), and the route's `/api/chat/sessions/{session_id}/stream` with `--field 'tags[copilot.stream.ttft_ms,number]'`.

4. `node scripts/diff-summaries.js results/<control>.json results/<candidate>.json` shows the two runs' client-side numbers side by side. It is a quick read: it reports the change, not whether the change is real.

perf-compare wants at least 200 samples from 30 traces per side and recommends 1000 requests per transaction; two baseline runs per side give that for every endpoint in the table except the rarest.

## Copilot and model spend

The workload sends no turn to a model unless told to.

- **The dummy engine.** The backend has a copilot engine that streams a fixed reply with no model behind it, chosen by `CHAT_TEST_MODE=true` on the copilot executor (it is read on every turn, so no restart of anything else is needed; the Sentry tag `copilot.engine` is `dummy`). It exercises the whole path users wait on: the route, the queue, the executor, the Redis relay and the stream. Nothing in this repository can set it on a preview; it is the infra repository's job, on the preview's copilot executor deployment.
- **`LOAD_TEST_COPILOT_ENGINE=dummy`** tells the run the target has it. `setup()` then sends one probe turn and checks the reply is the dummy's; if it is not, the copilot turn scenario is disabled for the run and the log says why. Without the variable, `copilot_turn` skips with a message and everything else runs.
- **`tests/copilot-real-model.js`** sends a capped number of turns at the real engine, for a reading of what users feel with a model behind the chat. It needs `LOAD_TEST_COPILOT_ENGINE=real`, caps the turns with `COPILOT_REAL_TURNS` (default 5, one at a time), prints the estimated cost before it starts (`COPILOT_REAL_COST_PER_TURN_USD`, default $0.10) and how far the account's daily usage meter moved after.

## Seeding

The target's marketplace is discovered, not created: a listing needs a creator profile, a submission and an admin's approval, which throwaway accounts cannot give. A preview's database holds only what its migrations and its five sign-in accounts put there, unless the seed fixture that `.github/workflows/platform-preview-seed-fixture.yml` bakes (it has approved listings) is restored into it, so a preview's marketplace is empty: the store journey reads the empty lists and skips the detail pages, the seed logs it and the summary's `coverage` says so. Everything else the journeys need, `setup()` creates under the run's own accounts, and only there:

- `SEED_GRAPHS` graphs (default 4) built from deterministic blocks, which also makes them library agents;
- `SEED_RUNS` runs of each (default 2), so the library and the executions list have history;
- one marketplace agent added to the library, when the store has one;
- `SEED_SESSIONS` chat sessions (default 6), with a turn in each when the dummy engine is confirmed.

The journeys read what the seed created: the library journey opens the run's agents, the graph journey runs them, the copilot journeys open the run's sessions. The seed counts are in the summary.

### Paywalled targets

With the `enable-platform-payment` flag on, an account without a subscription gets `402 A subscription is required` from graph execution and chat session creation, and a preview reads the flag the way dev does. A trial and every paid tier go through Stripe Checkout, whose hosted page k6 cannot drive, so the run completes the checkout the way the page would, in Stripe test mode:

- Export `STRIPE_TEST_SECRET_KEY`, a test-mode key (`sk_test_` or `rk_test_`) of the Stripe account the target uses; previews use the same test account as dev. `./k6.sh` passes it into the container by name. The tests check its prefix, send it to `api.stripe.com` and nowhere else, refuse a live key, refuse to use it against `PROD`, and never log it or write it to a summary.
- For each account, `setup()` asks the backend for a Checkout session for the cheapest tier it prices (`POST /credits/subscription`, which also creates the Stripe customer), then, through the Stripe test API, attaches the `pm_card_visa` test card to that customer and starts a subscription on the session's price: paid at once, no trial, tagged `autogpt_load_test_run=<RUN_ID>`. One `GET /chat/usage` then makes the backend copy the tier from Stripe (its lazy reconcile, which runs at most once per five minutes per user, so this is the account's first request), and `GET /credits/subscription` confirms it. About eight requests and a few seconds per account.
- `teardown()` cancels the subscriptions the run created; `seeding.subscriptions` and `seeding.subscriptions_cancelled` in the summary count them. A run that stops before its teardown leaves them behind: `./k6.sh run scripts/cancel-load-test-subscriptions.js` cancels every active subscription carrying the tag, or one run's with `RUN_FILTER=<RUN_ID>`.

Without the key, the seed notices on the first `402`: the graph journey and the copilot turn journey skip for the run, with the reason in the log; the copilot reads journey still runs, since the session list, usage, connections and experts routes answer for such accounts, though it has no session detail to open; and the summary says so under `coverage`, with `runs: 0` and `sessions: 0` under `seeding`. The other ways past the paywall are the target's: `FORCE_FLAG_ENABLE_PLATFORM_PAYMENT=false` on its REST server, executor and copilot executor (the flag is read in all three), or accounts with a subscription tier in `LOAD_TEST_ACCOUNTS` (see [Accounts on previews](#accounts-on-previews)).

## How authentication works

The backend accepts only the ES256 JWTs that the frontend's Better Auth issues (audience `authenticated`), and verifies them against `<frontend>/api/auth/jwks`. `lib/auth.js` gets them the way the browser does:

- `setup()` signs each account in once per run (`POST <frontend>/api/auth/sign-in/email`, or `sign-up/email` for throwaway accounts) and exchanges the session for a token (`GET <frontend>/api/auth/token`). Every VU then uses one account's token, so a run signs in once per account however many VUs it has.
- Better Auth lets one IP make 3 sign-in or sign-up requests, then refuses more until 10 seconds pass without one. `setup()` sends them in bursts of 3 with an 11-second pause, and waits out any `429`. Eight accounts take about 40 seconds.
- Tokens last an hour. Each VU fetches a new one with its account's session 10 to 20 minutes before the old one expires, so runs can last longer than an hour.
- Before the load starts, `setup()` sends one authenticated request and stops the run unless it succeeds.

Passwords, cookies and tokens stay in memory and are never logged.

## Accounts

### Accounts on previews

With `LOAD_TEST_ACCOUNTS` unset, `setup()` signs up `LOAD_TEST_ACCOUNT_COUNT` (default 8) throwaway accounts named `loadtest-<RUN_ID>-<n>@previews.agpt.co`, with random passwords that never leave the run. Previews allow sign-up for `@previews.agpt.co` and do not require email verification, so no email is sent and no shared secret is needed. The accounts and everything seeded under them stay in the preview's database until the preview is torn down. Copilot caps running turns per user, so more accounts spread the turns; keep at least eight for the stress profile.

To run as the seeded personas instead, set `LOAD_TEST_ACCOUNTS`. Every preview has `preview-admin`, `preview-existing`, `preview-clean`, `preview-pro` and `preview-enterprise` at `@previews.agpt.co`, sharing one password, the `PREVIEW_ACCOUNTS_PASSWORD` secret in the team Bitwarden. The preview seeder (`frontend/scripts/seed-preview-accounts.ts`) creates their sign-in and role only, not a subscription tier, so they meet the paywall like any new account until an admin grants them one (see [Paywalled targets](#paywalled-targets)). What a run seeds under them stays in the preview's database and adds up across runs, and a couple of accounts share copilot's per-user caps, so keep the load small:

```bash
export LOAD_TEST_ACCOUNTS="$(jq -n --arg p "$PREVIEW_ACCOUNTS_PASSWORD" \
  '[{email: "preview-pro@previews.agpt.co", password: $p},
    {email: "preview-enterprise@previews.agpt.co", password: $p}]')"
```

### Accounts on dev

Dev's sign-up allowlist does not include `@previews.agpt.co`, and `@agpt.co` addresses can only sign up through Google, so the tests cannot sign up on dev. Running there needs a few dedicated email-and-password accounts (an admin can create them with Better Auth's admin API, `POST https://dev-builder.agpt.co/api/auth/admin/create-user`, or a domain for them can be added to dev's `AUTH_SIGNUP_ALLOWLIST` in the infra repo), verified if dev requires email verification, with their passwords in the team secret store, handed to the tests as `LOAD_TEST_ACCOUNTS`. The whole team shares dev, so keep the load small (`LOAD_PROFILE=smoke` or `RATE_SCALE=0.2`) and tell the team first.

### `LOAD_TEST_ACCOUNTS`

A JSON array of accounts that already exist on the target:

```json
[{ "email": "loadtest-1@example.com", "password": "..." }]
```

The tests read it only from the environment. To keep it between runs, save it as `configs/accounts.json` (gitignored) and `export LOAD_TEST_ACCOUNTS="$(cat configs/accounts.json)"`. The tests sign in each account once; VUs share them round-robin, and the seed data is created under each.

### Local stack

`K6_ENVIRONMENT=LOCAL` targets `http://localhost:8006`, `http://localhost:3000` and `ws://localhost:8001/ws`; inside Docker, `./k6.sh` points them at `host.docker.internal`. The tests sign up throwaway accounts unless the local frontend restricts sign-up. A local stack with `CHAT_TEST_MODE=true` in its backend `.env` runs the dummy copilot engine.

## Environment variables

`./k6.sh` passes these into the container by name when they are set in your shell, so values never appear on a command line.

| Variable                                                                           | Used for                                                                                                                                                                                  |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `K6_ENVIRONMENT`                                                                   | `PREVIEW` (default), `DEV`, `LOCAL` or `PROD`                                                                                                                                             |
| `PR_NUMBER`                                                                        | The pull request whose preview `PREVIEW` targets                                                                                                                                          |
| `K6_BACKEND_URL`, `K6_FRONTEND_URL`, `K6_WS_URL`                                   | Override the target's URLs, for a stack on other ports or behind another name                                                                                                             |
| `LOAD_PROFILE`                                                                     | `smoke`, `baseline` (default) or `stress`                                                                                                                                                 |
| `SCENARIOS`                                                                        | A comma-separated subset of the journeys                                                                                                                                                  |
| `RATE_SCALE`, `WARMUP`, `DURATION`, `THRESHOLD_SCALE`                              | Adjust the profile                                                                                                                                                                        |
| `RUN_ID`                                                                           | Names the run's accounts and summary (default: a timestamp from `./k6.sh`)                                                                                                                |
| `RUN_SEED`                                                                         | Fixes the sequence of choices the journeys make (default fixed)                                                                                                                           |
| `SEED_GRAPHS`, `SEED_RUNS`, `SEED_SESSIONS`                                        | How much each account is seeded with (defaults 4, 2, 6)                                                                                                                                   |
| `LOAD_TEST_COPILOT_ENGINE`                                                         | `dummy` to send copilot turns at a target with `CHAT_TEST_MODE=true`; `real` for `tests/copilot-real-model.js` only                                                                       |
| `COPILOT_TURN_TIMEOUT`                                                             | The cap on one stream (default `120s` in the workload, `300s` for the real model)                                                                                                         |
| `COPILOT_REAL_TURNS`, `COPILOT_REAL_CONCURRENCY`, `COPILOT_REAL_COST_PER_TURN_USD` | `tests/copilot-real-model.js` only                                                                                                                                                        |
| `LOAD_TEST_ACCOUNTS`                                                               | Accounts to sign in with; required on `DEV` and `PROD`                                                                                                                                    |
| `LOAD_TEST_ACCOUNT_COUNT`                                                          | Throwaway accounts to sign up when `LOAD_TEST_ACCOUNTS` is unset (default 8, at most 100)                                                                                                 |
| `K6_ALLOW_PROD`                                                                    | Must be `true` for `PROD`; agree a production run with the team first                                                                                                                     |
| `K6_IMAGE`, `K6_REBUILD_IMAGE`                                                     | Run another k6 image, or rebuild the default one                                                                                                                                          |
| `K6_CLOUD_TOKEN`, `K6_CLOUD_STACK_ID`, `K6_CLOUD_PROJECT_ID`                       | Grafana Cloud k6 only, and passed in only with `--cloud`                                                                                                                                  |
| `STRIPE_TEST_SECRET_KEY`                                                           | A Stripe test-mode key of the target's Stripe account: the run subscribes its accounts past the paywall and cancels the subscriptions after (see [Paywalled targets](#paywalled-targets)) |
| `RUN_FILTER`                                                                       | `scripts/cancel-load-test-subscriptions.js` only: cancel one run's leftover subscriptions instead of every run's                                                                          |
| `GRAPH_RUN_TIMEOUT_MS`                                                             | How long the graph journey waits for a run to finish (default `60000`)                                                                                                                    |

## What it costs

A run on your machine costs nothing beyond the load it puts on the target: the preview's pods and database. With the dummy engine, copilot turns cost nothing; without `LOAD_TEST_COPILOT_ENGINE`, none are sent. Graph runs use deterministic blocks that charge no credits. The subscriptions a run creates with `STRIPE_TEST_SECRET_KEY` are Stripe test-mode objects: no money moves, and teardown cancels them.

Grafana Cloud k6 bills each test as its peak VUs times its whole duration, in virtual-user hours (VUh), whether it runs in the cloud (`k6 cloud run`) or streams results there (`--out cloud`). The Grafana Cloud org is on the Free plan, which allows 500 VUh a month; beyond that the paid plan costs about $0.15 per VUh on top of a $19 monthly platform fee. So run locally. `./k6.sh` refuses any command that would use Grafana Cloud unless `--cloud` is its first argument and `K6_CLOUD_TOKEN` is set, and passes no Grafana Cloud credentials into the container otherwise. An agreed cloud run needs a Grafana Cloud k6 API token and the stack's numeric ID; the project is `4254406`. Results appear at <https://significantgravitas.grafana.net/a/k6-app/>.

## Sentry

| Target  | Sentry project                                                     | Environment                      |
| ------- | ------------------------------------------------------------------ | -------------------------------- |
| Preview | `autogpt-server-previews` (backend), `builder-previews` (frontend) | `pr-<N>`                         |
| Dev     | `autogpt-server` (backend), `builder` (frontend)                   | `app:dev-behave:cloud` (backend) |

Each route is a transaction named by its path, such as `/api/store/skills/{slug}`. `/health` is not traced, and the sign-in requests from `setup()` go to the frontend project. The copilot executor's turn is its own transaction, `copilot.turn`, with the engine and timings as span data. `/api/chat/sessions/{session_id}/stream` and `/ws` are long-lived transactions that measure session length, not latency.

## Troubleshooting

- **`DEV needs LOAD_TEST_ACCOUNTS`**: dev and prod have no throwaway sign-up; see [Accounts on dev](#accounts-on-dev).
- **`Signing up throwaway account ... failed: HTTP 403`**: the target does not allow sign-up for `@previews.agpt.co`. Use `LOAD_TEST_ACCOUNTS`.
- **`Getting a token ... failed: HTTP 401` right after sign-up**: the target requires email verification, so a sign-up gets no session. Use verified accounts in `LOAD_TEST_ACCOUNTS`.
- **`... rejected the token from ...`**: the backend verifies tokens against a different frontend's JWKS than the one the test signed in at.
- **`The first authenticated request ... failed`**: the backend is down, still starting, or not where `K6_ENVIRONMENT` and `PR_NUMBER` point.
- **`seed: ... refused: HTTP 402 A subscription is required`**, then `graph_run: skipped` and `copilot_turn: skipped`: the target paywalls new accounts; export `STRIPE_TEST_SECRET_KEY`, see [Paywalled targets](#paywalled-targets).
- **`seed: subscribing account N in Stripe test mode failed: ...`**: the reason follows. `retrieving the checkout session failed` means the key belongs to another Stripe account than the target's; `the target reports tier NO_TIER` means the backend did not pick the subscription up, which happens when its reconcile already ran for that account within the last five minutes.
- **`STRIPE_TEST_SECRET_KEY is not a Stripe test-mode key`**: only `sk_test_` and `rk_test_` keys are accepted, and none against `PROD`.
- **`copilot_turn: skipped`**: set `LOAD_TEST_COPILOT_ENGINE=dummy` once the target runs `CHAT_TEST_MODE=true`; see [Copilot and model spend](#copilot-and-model-spend).
- **`copilot engine probe: the reply was not the dummy's`**: the target is not in test mode; the probe spent one real turn and the scenario is off for this run.
- **Copilot turns time out with no `finish`**: the target's copilot executor is not running or not consuming the queue; the route answers 200 and heartbeats until the stream's TTL.
- **`graph_run` never sees a completed run**: the executor is not running, or the WebSocket URL is wrong (`K6_WS_URL`); the journey falls back to polling and reports `graph_runs_failed`.
- **`dropped_iterations` fails**: k6 had no free VU when an iteration was due; the summary's `by_journey` says which journey. Each journey pre-allocates three times its expected concurrency (`configs/profiles.js`); a slower target needs more, so raise its `EXPECTED_ITERATION_SECONDS` there, or lower `RATE_SCALE`.
- **`429` during setup**: another sign-in from the same IP used up the burst. `setup()` waits and retries; it fails only after five refusals in a row.
- **`k6.sh: this command would use Grafana Cloud k6`**: run it on your machine instead.
- **`docker build` stalls on a credential helper**: `./k6.sh` builds with an empty temporary `DOCKER_CONFIG`; if your own `docker pull` hangs, do the same.
