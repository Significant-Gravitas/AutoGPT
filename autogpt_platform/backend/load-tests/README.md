# AutoGPT Platform Load Tests

[k6](https://grafana.com/docs/k6/latest/) tests that drive the platform's user-facing workload at a deployed target, a `!deploy` preview as a rule, and measure what users feel. The workload is centred on copilot and experts, the product's primary interaction points: how fast a chat opens and an open chat's polling answers, a turn's time to first token on the dummy engine, the whole SDK engine against a scripted model, and, opt-in, the real model; how fast the expert pages answer and a hire installs; and how fast the store, library and account pages answer and a graph run starts and finishes. The backend reports every request to Sentry, and the [perf-compare skill](../../../.claude/skills/perf-compare/SKILL.md) decides whether two builds differ.

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

   `setup()` runs before the load: signing up and subscribing the accounts, seeding them, hiring their experts, waiting for the seed runs and probing the copilot engines takes about ten minutes with the default eight accounts.

3. Run the baseline profile, the comparison run: a one-minute warm-up, then ten minutes at the rates below.

   ```bash
   LOAD_PROFILE=baseline ./k6.sh run tests/workload.js
   ```

Each run prints a report and writes `results/<RUN_ID>.json` (see [The summary](#the-summary)); the log says why any journey skipped (`K6_LOG_OUTPUT=file=results/<RUN_ID>.log` keeps it). The run signs up its own accounts and seeds its own data, and its teardown deletes the graph runs, library agents and chat sessions it created and archives the experts it hired (see [Seeding](#seeding)). Previews paywall new accounts, so for graph runs, chat sessions and turns export `STRIPE_TEST_SECRET_KEY`, a test-mode key of the Stripe account previews use: the run then subscribes its accounts in Stripe test mode and cancels the subscriptions at the end (see [Paywalled targets](#paywalled-targets)). Copilot turns run on the engine each account's address names: a preview that routes load-test accounts by address sends the dummy and scripted accounts' turns away from the real model, and `setup()` confirms that with a probe before it sends any turn; on a target that does not, the run fails after one probe turn, and `LOAD_TEST_COPILOT_TURNS=off` runs everything else (see [Copilot turns and model spend](#copilot-turns-and-model-spend)). Comment `!undeploy` when you are done with the preview.

## The workload

`tests/workload.js` runs up to eleven journeys at once, each as an open-model k6 scenario (`ramping-arrival-rate`): iterations start at a fixed rate whatever the target's latency, so a slower build cannot quietly send fewer requests. Queueing shows up as latency, and, if every virtual user is busy, as `dropped_iterations`, which fails the run. The real-model turns are the exception: a capped number, one at a time.

The rates are iterations per minute at the baseline profile, weighted by each family's share of user-facing requests in production (Sentry, `autogpt-server`, `app:prod-behave:cloud`, seven days, about 255,000 requests): copilot about 60%, store about 17%, account reads about 16%, experts about 2%, graphs and the library about 1%. In the run, copilot makes about 57% of the requests and the expert journeys about 9%, together two-thirds: experts get well above their raw share because they are a primary interaction point, and users work with them far more often than they hire them. Store (about 16%) and account reads (about 15%) follow production; rare but slow journeys get more than their share so a ten-minute run reaches a usable sample.

| Journey                 |                           Rate | What one iteration does                                                                                                                                                                                                                                                                                                                                                                    | Why this weight                                                                                                                                                                                                                                                                                         |
| ----------------------- | -----------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `copilot_reads`         |                             18 | Opens a chat: `GET /chat/sessions`, `/chat/usage`, `/chat/transports`, `/chat/connections`, `/experts/identities`, then one session's detail, its review cards (`/review/session/{id}`, `/review/session/{id}/answered`) and its queued messages (`/chat/sessions/{id}/messages/pending`)                                                                                                  | The chat's reads are production's largest family: review cards 75k requests a week, the session list 48k, usage 17k                                                                                                                                                                                     |
| `copilot_poll`          |                             24 | A chat left open on a review card, polled as the page polls it: five `GET /review/session/{id}` 2 s apart, one session list, and usage every third iteration                                                                                                                                                                                                                               | The page polls its cards every 2 s, the list every 10 s and usage every 30 s, which makes the card route production's largest                                                                                                                                                                           |
| `copilot_turn`          |                              9 | A turn on the dummy engine: `POST /chat/sessions` (30% of the time one of the account's seeded sessions instead), `POST /chat/sessions/{id}/stream` read to `finish`, then the session, usage and queued messages, and for a new chat the session list again, twice 2 s apart, for its title                                                                                               | The platform's own path without a model; about 850 streams a week in production, so the rate is a sample-size floor                                                                                                                                                                                     |
| `copilot_turn_scripted` |                              2 | A turn on the SDK engine against the scripted model, always on a new session, half of them with the account's expert: the script's marker and a short request, streamed to `finish`, then the session                                                                                                                                                                                      | The whole engine (sandbox, CLI, a tool round) at no model cost; each starts a sandbox, so a floor of twenty a run                                                                                                                                                                                       |
| `copilot_turn_real`     | `LOAD_TEST_COPILOT_REAL_TURNS` | Off unless asked for, at most 10 turns, one at a time through the steady phase: a new session and a one-sentence request on the real model                                                                                                                                                                                                                                                 | What users feel with a model behind the chat; about $0.07 a turn (see [Real-model turns](#real-model-turns))                                                                                                                                                                                            |
| `expert_work`           |                              4 | Works with the account's hired expert: `GET /experts`, `/experts/{id}`, `/experts/{id}/activity`, `/schedules`, `/home`, its chats (`/chat/sessions?expert_id=…`, twice) and files (`/workspace/files?expert_id=…&origin=generated`), then one of its chats (70% a seeded one, 30% a new one), a dummy turn in it, the chat and the session list                                           | Experts are about 2% of production's requests, but a primary interaction point, and users work with them far more often than they hire them                                                                                                                                                             |
| `expert_hire`           |                              1 | Hires from the marketplace: `GET /experts/templates`, `/experts`, `/integrations/providers/system`, `POST /experts`, the composer's reads, `POST /chat/sessions` with `expert_kickoff`, the kickoff turn (dummy) and the session, `GET /experts` every 2 s while the hire installs (at most 15 times); then deletes the kickoff chat and archives the expert, timed as their own endpoints | Rare but slow: the templates p50 1.3 s and an expert's page p95 3.0 s in production                                                                                                                                                                                                                     |
| `account_reads`         |                             10 | What every page load asks: `GET /credits`, `/credits/subscription`, `/credits/trial`, `/onboarding`, `/onboarding/completed`, `/integrations/credentials`, `/integrations/providers`, `/integrations/providers/system`, `/orgs`, `/auth/user/timezone`                                                                                                                                     | About 16% of requests, and `/credits/subscription` is the slowest common read in production                                                                                                                                                                                                             |
| `store_browse`          |                             15 | Browses the marketplace without signing in: `GET /store/skills`, a skill's detail, `GET /store/agents`, an agent's detail, a search, the creators list; then the account's `GET /store/profile`                                                                                                                                                                                            | Skills list and detail, store agents and search are about 17% of requests                                                                                                                                                                                                                               |
| `library_browse`        |                              4 | `GET /library/agents`, a library search, then one agent's detail                                                                                                                                                                                                                                                                                                                           | About 1% of requests with graphs, but p50 of 11 s for the list and 39 s for the detail in production                                                                                                                                                                                                    |
| `graph_run`             |                              3 | Opens one of the account's seeded graphs (`GET /graphs/{id}`), opens the WebSocket and subscribes to it, starts a run of it, waits for the run's events to say it completed, confirms with `GET /graphs/{id}/executions/{exec}`, and lists executions                                                                                                                                      | 130 to 210 runs a day in production, started by webhooks, schedules and copilot; this is the executor path, with the WebSocket users watch it on. A preview's executor is at capacity at about ten runs a minute, where a small slowdown queues runs without bound, so the baseline holds them at three |

The graphs are built from deterministic blocks that do real work without a model or credentials (lib/graphs.js), so a run costs nothing and finishes in a known number of steps. The journeys that read take whichever account their virtual user signed in as; the journeys that send turns take their engine's accounts one after the other (see [Account modes](#account-modes)).

`expert_hire` hires, on each account, the templates the target lists other than the seed's work template, one after the other, so a run never hires the same template on an account twice: a second hire of an archived expert's template revives that expert, a different path. An account that has hired every one it may hires no more for the rest of the run, which a long stress run reaches; the log says so.

### Profiles

| `LOAD_PROFILE` | Rates          | Warm-up | Steady | Thresholds  | Use                                                                                                                               |
| -------------- | -------------- | ------- | ------ | ----------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `smoke`        | 0.1 x baseline | none    | 1 min  | 3 x relaxed | Does the target work, does the seeding work                                                                                       |
| `baseline`     | the table      | 1 min   | 10 min | as declared | The comparison run; about 11 requests a second in all, below what a preview's one-CPU API server can serve (it saturates near 18) |
| `stress`       | 3 x baseline   | 1 min   | 10 min | 2 x relaxed | What breaks first                                                                                                                 |

The warm-up ramps each rate up from zero. Requests made during it carry `phase:warmup`, and so do its iterations: each iteration tags its VU with its phase, so k6's own `iteration_duration` splits by phase too. Every threshold, and every per-endpoint, per-journey and user-facing number in the summary, covers `phase:steady` only, with one exception: `dropped_iterations`, which k6 counts against the scenario without a phase, covers the whole run, and a drop in the warm-up, which offers less than the steady rate, fails it too. The real-model turns start when the warm-up ends, so all of them are steady.

Settings: `SCENARIOS=copilot_turn,store_browse` runs a subset; `RATE_SCALE=0.5` multiplies every rate; `WARMUP` and `DURATION` replace the phase lengths; `THRESHOLD_SCALE` relaxes the thresholds; `RUN_SEED` fixes which items each iteration picks (default fixed, so two runs send the same sequence); `LOAD_TEST_ACCOUNT_COUNT` is the number of accounts (default 8).

### What is measured

- Every request carries `endpoint` (such as `store.skill_detail`) and `phase` tags, and one check named after the endpoint. Thresholds in `configs/thresholds.js` set a p95 per endpoint, above production's p95 for the route, and a failure rate under 2%; each number there says where it comes from. The graph run's own numbers, its execute route and its WebSocket are sized on a preview instead, whose executor is smaller than production's: a run of a seeded graph takes about 30 s there. The hire, the archive and the two session creates are too rare in production for a p95 of their own and get bounds of a few seconds; `expert_hire`'s deletes of its kickoff chats are timed as `copilot.session_delete`, apart from the reads, and judged only by the run's overall failure rate.
- Custom Trends for what users feel: `copilot_first_chunk_ms` (the first frame), `copilot_ttft_ms` (the first text), `copilot_turn_ms` (the `finish` frame), `copilot_chunks_per_turn`, each tagged with the turn's `engine` and `turn` (`chat`, or `kickoff` for a hire's first turn) and judged per engine: the dummy engine's chat turns within 2 s, 3 s and 10 s at p95, a hire's kickoff within 2 s to the first frame and 75 s to the finish (the executor holds a turn on a new expert's chat until the expert's setup is done, for up to a minute), the scripted engine's within 5 s, 30 s and 60 s (starting bounds, to tighten to what previews measure), the real model's reported only. `graph_run_start_ms` (execute to the first running event) and `graph_run_ms` (execute to completed); `ws_connect_ms` and `ws_event_latency_ms` (the event's `ended_at` to its arrival). Counters tell completed, queued and failed copilot turns, per engine, and graph runs apart; failed turns may be one in twenty on the dummy engine and one in ten on the scripted one.
- A graph run with no terminal event within `GRAPH_RUN_TIMEOUT_MS` (60 s) is polled for the rest of that timeout and at least `GRAPH_RUN_POLL_MS` (30 s) more, so a run that finishes just after it counts as completed, with its real duration; one that never finishes counts in `graph_run_errors`. A run whose socket failed (it did not open, the server refused the subscription or dropped the connection) counts in `ws_errors` and goes on by polling, tagged `source:poll`, so `graph_run_errors` measures the executor and `ws_errors` the WebSocket server. The socket's `name` and `url` tags are its URL without the query.
- `copilot.stream`'s `http_req_duration` is the whole stream, so it has no latency threshold; the Trends above are its numbers.
- An iteration that takes two minutes has stalled; a scripted turn's iteration gets its stream timeout plus half a minute, and the real-model turns' are reported only.
- `coverage_failures` counts the journeys `setup()` found broken on the target (see [The summary](#the-summary)); its threshold, `count==0`, fails the run, since a journey switched off has no samples to fail its own thresholds with.

### The summary

`handleSummary` writes `results/<RUN_ID>.json` with:

- `run_id`, `profile`, `seed`, the scenarios with the rate each ran at (or, for the real-model turns, how many), its requests per iteration and why it has its weight;
- `target`: name, URLs and the Sentry project and environment to query (`autogpt-server-previews`, `pr-<N>`); the name says when `K6_*_URL` overrode a URL, and the Sentry entry is empty when the backend URL was overridden; `target.build` is the backend's `/health` answer, which does not name the commit, so select the preview's traces by environment and window (its Sentry `release` is the git sha);
- `window`, UTC: `started_at`; `load_started_at`, after setup; `steady_started_at` and `steady_ended_at`, the steady phase, which is what the thresholds judge and what perf-compare's `window=` should select; and `ended_at`, after teardown;
- per endpoint and per custom metric: count, p50, p95, p99, failure rate, steady phase only; a turn's metrics once per engine, such as `copilot_ttft_ms{engine:dummy}`;
- `by_journey`: each journey's steady-phase iteration time, and its dropped iterations over the whole run;
- the seeding counts: what the seed created, including the subscriptions created and cancelled, the experts hired and their chats, and how many store agents, skills and expert templates the target had, and what teardown deleted and archived;
- `coverage`: per journey, `covered`, `partial`, `skipped` or `failed`, why when it is not covered, and how many of the run's accounts could run it. `partial` and `skipped` are limits of the target or the settings: the paywall without a Stripe key, an empty marketplace, `LOAD_TEST_COPILOT_TURNS=off`, no account of a journey's mode, no expert template left to hire. `failed` is the target failing what the run needed and asked for: no seed run completed, an engine failed its probe, an account stayed paywalled although the run had a key, the seed could not create a graph, a library agent or a chat session or hire an expert, the target lists no expert templates, the store's agents or skills list answered with an error (a skills `404` or an empty list is a limit); it fails the run through `coverage_failures`. The text report prints the same under `NOT COVERED IN FULL`, because a list read against an empty marketplace is fast for the wrong reason;
- `cost`, when the run sent real-model turns: how many it planned and sent, and the estimate;
- every threshold's result.

`RUN_ID` comes from `./k6.sh` (a timestamp) unless you set it. The summary and the run's accounts are named after it.

## Comparing two builds

The verdict comes from the backend's traces in Sentry through perf-compare, not from k6's client-side numbers: perf-compare resamples by trace, corrects for multiple comparisons and refuses to judge on too little data. The procedure:

1. Deploy the candidate with `!deploy`, and deploy a control the same way: a pull request at the dev commit the candidate branches from, with no changes. Each preview gets its own namespace and database, so they are alike.
2. Run the baseline profile at both, interleaved, with the same seed. `scripts/ab.sh` does this (A B B A by default, about 90 minutes: a baseline run takes about 22, of which about ten are setup) and prints the perf-compare command for the steady-phase windows it produced. Runs whose `coverage` differs did not send the same load, so then it prints their coverage instead, and no command:

   ```bash
   scripts/ab.sh <control PR> <candidate PR>
   ```

   By hand, the equivalent is four `LOAD_PROFILE=baseline ./k6.sh run tests/workload.js` runs alternating `PR_NUMBER`, taking each run's `steady_started_at` and `steady_ended_at` from its summary and checking that their `coverage` matches.

3. Wait a few minutes for Sentry to ingest the last events, then name the transaction you care about and compare:

   ```bash
   S=$(git rev-parse --show-toplevel)/.claude/skills/perf-compare/scripts/perf_compare.py
   python3 "$S" --project autogpt-server-previews --transaction '/api/store/skills/{slug}' \
     --baseline env=pr-<CONTROL> window=<START1>/<END1> window=<START4>/<END4> \
     --candidate env=pr-<CANDIDATE> window=<START2>/<END2> window=<START3>/<END3>
   ```

   For copilot, compare the executor's `copilot.turn` transaction and its span data: `--transaction copilot.turn --field 'tags[copilot.turn.ttft_ms,number]'` (also `copilot.turn.queue_wait_ms`, `copilot.turn.duration_ms`), and the route's `/api/chat/sessions/{session_id}/stream` with `--field 'tags[copilot.stream.ttft_ms,number]'`. The run's dummy and scripted turns land there, marked as load-test turns with their engine; a window of normal use on dev or production gives the same fields for the real model, with `--project autogpt-server` and the environment from the [Sentry](#sentry) table.

4. `node scripts/diff-summaries.js results/<control>.json results/<candidate>.json` shows the two runs' client-side numbers side by side. It is a quick read: it reports the change, not whether the change is real.

perf-compare wants at least 200 samples from 30 traces per side and recommends 1000 requests per transaction; two baseline runs per side give that for every endpoint in the table except the rarest.

## Copilot turns and model spend

The workload sends turns at three engines, each from its own accounts, so the load costs nothing in model spend except for the few real-model turns it is asked for. The engine is the account's (see [Account modes](#account-modes)):

- **The dummy engine** streams a fixed reply, `I counted: 1... 2... 3. All done!`, with no model, sandbox or memory work behind it. It exercises the platform's own path, the one users wait on before any model answers: the route, the queue, the executor, the Redis relay and the stream. `copilot_turn`, `expert_work`'s turn and `expert_hire`'s kickoff run on it.
- **The scripted model** plays the provider for the SDK engine, so the whole engine runs, the sandbox, the CLI and the tool loop, at no model cost (see [The scripted model](#the-scripted-model)). `copilot_turn_scripted` runs on it.
- **The real model** runs only `copilot_turn_real`'s capped turns (see [Real-model turns](#real-model-turns)).

Before it sends any turn, `setup()` probes. One dummy account's turn on a seeded session must come back with the dummy's reply; only then does one scripted account's turn on a new session, marked `[[script:loadtest-turn#probe-<RUN_ID>]]`, have to end with the script's final text. A dummy probe that fails switches every turn off, the dummy, scripted and real journeys, `expert_work`'s turn and `expert_hire`'s kickoff, and no scripted or real turn is sent: a target that does not route the accounts by address answered the probe with its real model, and that one turn is all the run spends. A scripted probe that fails switches off `copilot_turn_scripted` only. Either fails the run, with the reason in the log and under `coverage`. `LOAD_TEST_COPILOT_TURNS=off` sends no turn at all, probes included: the turn journeys skip, the expert journeys still make their reads, and the run passes or fails on the rest. Copilot caps a user's running turns at 5 and turns in flight at 15; the turn journeys take their accounts one after the other by iteration, so the load, not the cap, sets the pace.

`tests/copilot-real-model.js` sends a capped number of turns at the real engine on their own, for a reading of what users feel with a model behind the chat outside the workload's mix. It needs `LOAD_TEST_COPILOT_ENGINE=real`, signs up real-model accounts only (and refuses accounts in `LOAD_TEST_ACCOUNTS` named for another engine), sends `COPILOT_REAL_TURNS` turns (a whole number from 1 to 50, default 5) `COPILOT_REAL_CONCURRENCY` at a time (1 to 5, default 1) from as many accounts as it runs at once unless `LOAD_TEST_ACCOUNT_COUNT` says otherwise, and does not subscribe them, so on a paywalled preview use `copilot_turn_real` instead; prints the estimated cost before it starts (`COPILOT_REAL_COST_PER_TURN_USD`, default $0.07) and how far the account's daily usage meter moved after.

### The scripted model

[`autogpt_platform/scripted_model`](../../scripted_model/README.md) plays the model provider from a script. A preview that runs it has `CHAT_LOAD_TEST_MODEL_URL` pointing at it, and the scripted accounts' CLI talks to it instead of the deployment's provider; the infra repo deploys it per preview with the scripts in [model-scripts/](model-scripts/), mounted at `/scripts`.

The run's script, [model-scripts/loadtest-turn.json](model-scripts/loadtest-turn.json), is one turn with one tool round: a short answer and a `bash_exec` call that runs `echo load-test` in the session's sandbox, then, once the command's output is back, a short final answer. Its usage is a few hundred tokens, so the account's allowance meter barely moves, and its text streams like a model's, three words every 30 ms. A message picks it with `[[script:loadtest-turn#<RUN_ID>-<iteration>]]` followed by one of the short requests. The scripted model answers turn n of a script to a conversation that already holds n assistant messages, so every scripted turn starts a new session; a request it cannot answer (no marker, an unknown script, a turn past the script's end) fails the turn with a 400, counted in `copilot_turn_errors{engine:scripted}`. The probe reads the final text from the same file, so an edited script is checked against itself once the preview deploys it.

### Real-model turns

`LOAD_TEST_COPILOT_REAL_TURNS=N` (at most 10, default 0) adds `copilot_turn_real`: N turns from the run's real-model account (`LOAD_TEST_REAL_ACCOUNTS`, default 1 when N is set), one at a time, from the end of the warm-up until the turns are sent or the steady phase ends, each a new session with a one-sentence request and a 300 s stream timeout. They cost money: `setup()` prints the estimate before anything is sent, N × `COPILOT_REAL_COST_PER_TURN_USD` (default $0.07, production's median for a one-step turn), and the summary's `cost` reports the turns sent and their estimate. They are sent only after the dummy probe passes. There are too few of them for a threshold, so their timings, error rate and dropped iterations are reported and never judged.

## Seeding

The target's marketplace and expert roster are discovered, not created: a listing needs a creator profile, a submission and an admin's approval, which throwaway accounts cannot give, and the roster's templates come from the skills catalog. A preview's database holds only what its migrations and its five sign-in accounts put there, unless the seed fixture that `.github/workflows/platform-preview-seed-fixture.yml` bakes (it has approved listings) is restored into it, so a preview's marketplace may be empty: the store journey reads the empty lists and skips the detail pages, the seed logs it and the summary's `coverage` says so. A target that lists no expert templates fails a run with expert journeys. Everything else the journeys need, `setup()` creates under the run's own accounts, and only there:

- `SEED_GRAPHS` graphs (default 4) built from deterministic blocks, which also makes them library agents;
- `SEED_RUNS` runs of each (default 2, at least 1), so the library and the executions list have history; the graph journey runs only once one of them has completed;
- one marketplace agent added to the library, when the store has one;
- `SEED_SESSIONS` chat sessions (default 6), with a turn in each of a dummy account's first three once the dummy engine is confirmed;
- for each dummy and scripted account, the expert template `LOAD_TEST_EXPERT_TEMPLATE` (an id or a name; default the first the target lists) hired, the hire's setup waited for, every account's at once and up to two minutes, and two chats with that expert.

The journeys read what the seed created: the library journey opens the run's agents, the graph journey runs them, the copilot journeys open the run's sessions, the expert journeys the run's experts and their chats. An account the seed could not get something for (a `402`, a failed request) sits out the journeys that need it, and the summary's `coverage` says how many accounts ran each. The seed counts are in the summary.

`teardown()` then cancels the run's Stripe test-mode subscriptions (see [Paywalled targets](#paywalled-targets)) and deletes what the run created, through the routes the app deletes with: every run of the seeded graphs (the seed's and the graph journey's), the library agents the seed created or added, the seeded chat sessions and expert chats, and, when a journey that starts sessions ran, every chat session the accounts started during the run, the probes' included. That last sweep covers only accounts that exist for load tests: the ones the run signed up, and accounts in `LOAD_TEST_ACCOUNTS` named for a load-test engine. On any other account there, which someone could be using at the same time, the run deletes only the sessions it seeded and leaves the ones its journeys started. Last it archives, as a user does, every expert the run hired: on each account, every active expert that was not there before the run, which covers the seed's work experts and any hire `expert_hire` did not archive itself. The backend soft-deletes runs and library agents, as it does for a user, deletes chat sessions outright, and keeps an archived expert with its schedules paused. A request the target cannot answer for the moment (no answer, or `502` to `504` while a pod restarts) is sent again a few times before the delete counts as failed; the summary counts the deletes, the archives and the failures. The rest stays:

- the graphs: the backend does not delete a graph that a library entry refers to, and a deleted library agent still does;
- the archived experts, with what their setup installed;
- the onboarding steps the seed completed and the credits they granted;
- the Stripe test-mode customers the backend created (teardown cancels their subscriptions);
- the throwaway accounts.

On a preview they go with the preview. Under `LOAD_TEST_ACCOUNTS` they add up across runs: a few graphs per account per run, hidden from its library, and an archived expert for each template hired. An account there keeps the experts it had before the run: the seed hires for it only after reading them, `expert_hire` hires only templates it has none of, and teardown archives none of them.

### Paywalled targets

With the backend's `ENABLE_PLATFORM_PAYMENT` setting on, an account without a subscription gets `402 A subscription is required` from graph execution and chat session creation. Dev turns it on, and a preview takes dev's setting. A trial and every paid tier go through Stripe Checkout, whose hosted page k6 cannot drive, so the run completes the checkout the way the page would, in Stripe test mode:

- Export `STRIPE_TEST_SECRET_KEY`, a test-mode key (`sk_test_` or `rk_test_`) of the Stripe account the target uses; previews use the same test account as dev. `./k6.sh` passes it into the container by name. The tests check its prefix, send it to `api.stripe.com` and nowhere else, refuse a live key, refuse to use it against production, and never log it or write it to a summary.
- For each account, whatever its mode, `setup()` asks the backend for a Checkout session for the cheapest tier it prices monthly (`POST /credits/subscription`, which also creates the Stripe customer), then, through the Stripe test API, attaches the `pm_card_visa` test card to that customer and starts a subscription on the session's price: paid at once, no trial, tagged `autogpt_load_test_run=<RUN_ID>`. One `GET /chat/usage` then makes the backend copy the tier from Stripe (its lazy reconcile, which runs at most once per five minutes per user, so this is the account's first request), and `GET /credits/subscription` confirms it. About eight requests and a few seconds per account.
- `teardown()` cancels the subscriptions the run created, before its deletes; `seeding.subscriptions` and `seeding.subscriptions_cancelled` in the summary count them. A `setup()` that throws cancels the ones it made, and deletes what it seeded, before it stops the run. A run stopped before its teardown, or a `setup()` that runs out of time, leaves them behind: `./k6.sh run scripts/cancel-load-test-subscriptions.js` cancels every active subscription carrying the tag, or one run's with `RUN_FILTER=<RUN_ID>`, and fails when it cannot list them or a cancellation fails, so run it again until it passes.

Without the key, each account meets the paywall at its first graph and its first chat session: the graph journey and the turn and expert journeys skip for the run, with the reason in the log; the library and copilot reads journeys still run, since the library, session list, usage, connections and experts routes answer for such accounts, though they have no agent or session detail to open; and the summary says so under `coverage`, with `graphs: 0`, `runs: 0` and `sessions: 0` under `seeding`. Accounts in `LOAD_TEST_ACCOUNTS` that already have a subscription tier get past the paywall too (see [Accounts on previews](#accounts-on-previews)). With the key, an account that stays paywalled fails the run: the subscription did not take, and the log says why.

## How authentication works

The backend accepts only the ES256 JWTs that the frontend's Better Auth issues (audience `authenticated`), and verifies them against `<frontend>/api/auth/jwks`. `lib/auth.js` gets them the way the browser does:

- `setup()` signs each account in once per run (`POST <frontend>/api/auth/sign-in/email`, or `sign-up/email` for throwaway accounts) and exchanges the session for a token (`GET <frontend>/api/auth/token`). Every VU then uses those tokens, so a run signs in once per account however many VUs it has.
- Better Auth lets one IP make 3 sign-in or sign-up requests, then refuses more until 10 seconds pass without one. `setup()` sends them in bursts of 3 with an 11-second pause, and waits out any `429`. Eight accounts take about 40 seconds.
- Tokens last an hour. Each VU fetches a new one for an account with that account's session 10 to 20 minutes before the old one expires, so runs can last longer than an hour.
- Before the load starts, `setup()` sends one authenticated request and stops the run unless it succeeds.

Passwords, cookies and tokens stay in memory and are never logged.

## Accounts

### Account modes

Each account's address names the engine its copilot turns run on. A target with `CHAT_LOAD_TEST_EMAIL_DOMAIN` set to the accounts' domain (`previews.agpt.co` on a preview) routes them by the address's prefix:

| Mode       | Throwaway address                                 | Engine                                                                                   | Turn journeys                                |
| ---------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------- |
| `dummy`    | `loadtest-dummy-<RUN_ID>-<n>@previews.agpt.co`    | The dummy engine: a canned reply, no model                                               | `copilot_turn`, `expert_work`, `expert_hire` |
| `scripted` | `loadtest-scripted-<RUN_ID>-<n>@previews.agpt.co` | The SDK engine, sandbox, CLI and tool loop included, against the scripted model          | `copilot_turn_scripted`                      |
| `real`     | `loadtest-<RUN_ID>-<n>@previews.agpt.co`          | The account's usual engine and the real model, as for any address without those prefixes | `copilot_turn_real`                          |

The run signs up `LOAD_TEST_SCRIPTED_ACCOUNTS` scripted accounts (default 2), `LOAD_TEST_REAL_ACCOUNTS` real ones (default 1 when `LOAD_TEST_COPILOT_REAL_TURNS` asks for turns, else 0), and makes the rest of `LOAD_TEST_ACCOUNT_COUNT` dummy accounts, at least one, numbered across the run: by default dummy 1 to 6 and scripted 7 and 8. A small `LOAD_TEST_ACCOUNT_COUNT` shrinks the defaults to leave a dummy account. An account in `LOAD_TEST_ACCOUNTS` gets the mode its address asks for, read the same way, and those two settings do not apply. The journeys that read, and graph runs, use every account; a turn journey sends turns only from its mode's accounts, and the expert journeys run as dummy accounts. A run with real accounts refuses a `RUN_ID` starting with `dummy-` or `scripted-`, since it would make their addresses read as another mode.

### Accounts on previews

With `LOAD_TEST_ACCOUNTS` unset, `setup()` signs up `LOAD_TEST_ACCOUNT_COUNT` (default 8) throwaway accounts named for their mode (see [Account modes](#account-modes)), with random passwords that never leave the run. Previews allow sign-up for `@previews.agpt.co` and do not require email verification, so no email is sent and no shared secret is needed. The accounts, and what teardown cannot delete under them (see [Seeding](#seeding)), stay in the preview's database until the preview is torn down. Copilot caps running turns per user, so more accounts spread the turns; keep at least eight for the stress profile.

To run as the seeded personas instead, set `LOAD_TEST_ACCOUNTS`. Every preview has `preview-admin`, `preview-existing`, `preview-clean`, `preview-pro` and `preview-enterprise` at `@previews.agpt.co`, sharing one password, the `PREVIEW_ACCOUNTS_PASSWORD` secret in the team Bitwarden. Their addresses name no load-test engine, so their turns would go to the real model: with no dummy account to probe the engine with, the turn and expert journeys skip and no turn is sent. The preview seeder (`frontend/scripts/seed-preview-accounts.ts`) creates their sign-in and role only, not a subscription tier, so they meet the paywall like any new account until an admin grants them one (see [Paywalled targets](#paywalled-targets)). What teardown cannot delete under them stays in the preview's database and adds up across runs (see [Seeding](#seeding)), and a couple of accounts share copilot's per-user caps, so keep the load small:

```bash
export LOAD_TEST_ACCOUNTS="$(jq -n --arg p "$PREVIEW_ACCOUNTS_PASSWORD" \
  '[{email: "preview-pro@previews.agpt.co", password: $p},
    {email: "preview-enterprise@previews.agpt.co", password: $p}]')"
```

### Accounts on dev

Dev's sign-up allowlist does not include `@previews.agpt.co`, and `@agpt.co` addresses can only sign up through Google, so the tests cannot sign up on dev. Running there needs a few dedicated email-and-password accounts (an admin can create them with Better Auth's admin API, `POST https://dev-builder.agpt.co/api/auth/admin/create-user`, or a domain for them can be added to dev's `AUTH_SIGNUP_ALLOWLIST` in the infra repo), verified if dev requires email verification, with their passwords in the team secret store, handed to the tests as `LOAD_TEST_ACCOUNTS`. Dev routes no account by address unless its `CHAT_LOAD_TEST_EMAIL_DOMAIN` names theirs, so name them for the dummy engine only if it does, or run with `LOAD_TEST_COPILOT_TURNS=off`. The whole team shares dev, so keep the load small (`LOAD_PROFILE=smoke` or `RATE_SCALE=0.2`) and tell the team first.

### `LOAD_TEST_ACCOUNTS`

A JSON array of accounts that already exist on the target:

```json
[{ "email": "loadtest-1@example.com", "password": "..." }]
```

The tests read it only from the environment. To keep it between runs, save it as `configs/accounts.json` (gitignored) and `export LOAD_TEST_ACCOUNTS="$(cat configs/accounts.json)"`. The tests sign in each account once; the reading VUs share them round-robin, the turn journeys take their mode's, and the seed data is created under each.

### Local stack

`K6_ENVIRONMENT=LOCAL` targets `http://localhost:8006`, `http://localhost:3000` and `ws://localhost:8001/ws`; inside Docker, `./k6.sh` points them at `host.docker.internal`. The tests sign up throwaway accounts unless the local frontend restricts sign-up. A local backend routes the accounts by address with `CHAT_LOAD_TEST_EMAIL_DOMAIN=previews.agpt.co` in its `.env`; the scripted accounts also need `CHAT_LOAD_TEST_MODEL_URL` pointing at a scripted model that serves this folder's `model-scripts/` (the scripted model's README says how to run one on its own). `CHAT_TEST_MODE=true` instead runs the dummy engine for every account: the dummy probe passes and the scripted one fails, so run with `LOAD_TEST_SCRIPTED_ACCOUNTS=0` there.

## Environment variables

`./k6.sh` passes these into the container by name when they are set in your shell, so values never appear on a command line.

| Variable                                                     | Used for                                                                                                                                                                                  |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `K6_ENVIRONMENT`                                             | `PREVIEW` (default), `DEV`, `LOCAL` or `PROD`                                                                                                                                             |
| `PR_NUMBER`                                                  | The pull request whose preview `PREVIEW` targets                                                                                                                                          |
| `K6_BACKEND_URL`, `K6_FRONTEND_URL`, `K6_WS_URL`             | Override the target's URLs, for a stack on other ports or behind another name; the target's name then says so, and a URL on one of production's hosts needs `K6_ALLOW_PROD=true`          |
| `LOAD_PROFILE`                                               | `smoke`, `baseline` (default) or `stress`                                                                                                                                                 |
| `SCENARIOS`                                                  | A comma-separated subset of the journeys                                                                                                                                                  |
| `RATE_SCALE`, `WARMUP`, `DURATION`, `THRESHOLD_SCALE`        | Adjust the profile                                                                                                                                                                        |
| `RUN_ID`                                                     | Names the run's accounts and summary (default: a timestamp from `./k6.sh`)                                                                                                                |
| `RUN_SEED`                                                   | Fixes the sequence of choices the journeys make (default fixed)                                                                                                                           |
| `SEED_GRAPHS`, `SEED_RUNS`, `SEED_SESSIONS`                  | How much each account is seeded with (defaults 4, 2, 6; `SEED_RUNS` at least 1)                                                                                                           |
| `LOAD_TEST_COPILOT_TURNS`                                    | `off` sends no copilot turn, probes included, for a target that does not route accounts by address (default `on`)                                                                         |
| `LOAD_TEST_SCRIPTED_ACCOUNTS`, `LOAD_TEST_REAL_ACCOUNTS`     | How many of the throwaway accounts are scripted (default 2) and real (default 1 with real-model turns, else 0); the rest are dummy accounts                                               |
| `LOAD_TEST_COPILOT_REAL_TURNS`                               | Real-model turns the workload sends, at most 10 (default 0: none, and no `copilot_turn_real`)                                                                                             |
| `COPILOT_REAL_COST_PER_TURN_USD`                             | The cost estimate per real-model turn (default `0.07`), in the workload and `tests/copilot-real-model.js`                                                                                 |
| `LOAD_TEST_EXPERT_TEMPLATE`                                  | The expert template, by id or name, each dummy and scripted account hires for `expert_work` and the scripted expert chats (default the first the target lists)                            |
| `COPILOT_TURN_TIMEOUT`                                       | The cap on one stream: the dummy turns in the workload (default `120s`), the real ones in `tests/copilot-real-model.js` (default `300s`)                                                  |
| `COPILOT_SCRIPTED_TURN_TIMEOUT`                              | The cap on one scripted turn's stream, the probe's included (default `180s`)                                                                                                              |
| `LOAD_TEST_COPILOT_ENGINE`                                   | `real` to run `tests/copilot-real-model.js`; the workload picks engines by account                                                                                                        |
| `COPILOT_REAL_TURNS`, `COPILOT_REAL_CONCURRENCY`             | `tests/copilot-real-model.js` only                                                                                                                                                        |
| `LOAD_TEST_ACCOUNTS`                                         | Accounts to sign in with; required on `DEV` and `PROD`                                                                                                                                    |
| `LOAD_TEST_ACCOUNT_COUNT`                                    | Throwaway accounts to sign up when `LOAD_TEST_ACCOUNTS` is unset (default 8, at most 100)                                                                                                 |
| `K6_ALLOW_PROD`                                              | Must be `true` for `PROD`, and for a `K6_*_URL` on one of production's hosts whatever the target; agree a production run with the team first                                              |
| `K6_IMAGE`, `K6_REBUILD_IMAGE`                               | Run another k6 image, or rebuild the default one                                                                                                                                          |
| `K6_CLOUD_TOKEN`, `K6_CLOUD_STACK_ID`, `K6_CLOUD_PROJECT_ID` | Grafana Cloud k6 only, and passed in only with `--cloud`                                                                                                                                  |
| `STRIPE_TEST_SECRET_KEY`                                     | A Stripe test-mode key of the target's Stripe account: the run subscribes its accounts past the paywall and cancels the subscriptions after (see [Paywalled targets](#paywalled-targets)) |
| `RUN_FILTER`                                                 | `scripts/cancel-load-test-subscriptions.js` only: cancel one run's leftover subscriptions instead of every run's                                                                          |
| `GRAPH_RUN_TIMEOUT_MS`                                       | How long the graph journey waits for a run's terminal event over the WebSocket (default `60000`)                                                                                          |
| `GRAPH_RUN_POLL_MS`                                          | How long, at least, the graph journey then polls a run that has not finished (default `30000`)                                                                                            |

## What it costs

A run on your machine costs nothing beyond the load it puts on the target: the preview's pods and database, plus, per scripted turn, a sandbox and the new session's title, which the deployment's own title model writes, a fraction of a cent. Dummy turns cost nothing, and scripted turns spend nothing on their replies. Real-model turns are sent only with `LOAD_TEST_COPILOT_REAL_TURNS`, about $0.07 each, and the run prints the estimate before it starts; a target that does not route the accounts by address spends one real turn on the dummy probe, and nothing more. Graph runs use deterministic blocks that charge no credits. The subscriptions a run creates with `STRIPE_TEST_SECRET_KEY` are Stripe test-mode objects: no money moves, and teardown cancels them.

Grafana Cloud k6 bills each test as its peak VUs times its whole duration, in virtual-user hours (VUh), whether it runs in the cloud (`k6 cloud run`) or streams results there (`--out cloud`). The Grafana Cloud org is on the Free plan, which allows 500 VUh a month; beyond that the paid plan costs about $0.15 per VUh on top of a $19 monthly platform fee. So run locally. `./k6.sh` refuses any command that would use Grafana Cloud unless `--cloud` is its first argument and `K6_CLOUD_TOKEN` is set, and passes no Grafana Cloud credentials into the container otherwise. An agreed cloud run needs a Grafana Cloud k6 API token and the stack's numeric ID; the project is `4254406`. Results appear at <https://significantgravitas.grafana.net/a/k6-app/>.

## Sentry

| Target  | Sentry project                                                     | Environment                      |
| ------- | ------------------------------------------------------------------ | -------------------------------- |
| Preview | `autogpt-server-previews` (backend), `builder-previews` (frontend) | `pr-<N>`                         |
| Dev     | `autogpt-server` (backend), `builder` (frontend)                   | `app:dev-behave:cloud` (backend) |

Each route is a transaction named by its path, such as `/api/store/skills/{slug}`. `/health` is not traced, and the sign-in requests from `setup()` go to the frontend project. The copilot executor's turn is its own transaction, `copilot.turn`, with the engine and timings as span data; a target that routes load-test accounts marks their turns with `copilot.load_test`. `/api/chat/sessions/{session_id}/stream` and `/ws` are long-lived transactions that measure session length, not latency.

## Troubleshooting

- **`DEV needs LOAD_TEST_ACCOUNTS`**: dev and prod have no throwaway sign-up; see [Accounts on dev](#accounts-on-dev).
- **`Signing up throwaway account ... failed: HTTP 403`**: the target does not allow sign-up for `@previews.agpt.co`. Use `LOAD_TEST_ACCOUNTS`.
- **`Getting a token ... failed: HTTP 401` right after sign-up**: the target requires email verification, so a sign-up gets no session. Use verified accounts in `LOAD_TEST_ACCOUNTS`.
- **`... rejected the token from ...`**: the backend verifies tokens against a different frontend's JWKS than the one the test signed in at.
- **`The first authenticated request ... failed`**: the backend is down, still starting, or not where `K6_ENVIRONMENT` and `PR_NUMBER` point.
- **`seed: account N cannot create graphs: HTTP 402 A subscription is required`** or **`seed: creating chat sessions for account N refused: HTTP 402 ...`**, then `graph_run: skipped` and `copilot_turn: skipped`: the target paywalls new accounts; export `STRIPE_TEST_SECRET_KEY`, see [Paywalled targets](#paywalled-targets).
- **`seed: subscribing account N in Stripe test mode failed: ...`**: the reason follows. `retrieving the checkout session failed` means the key belongs to another Stripe account than the target's; `the target reports tier NO_TIER` means the backend did not pick the subscription up, which happens when its reconcile already ran for that account within the last five minutes. If that account then meets the paywall, the run fails.
- **`STRIPE_TEST_SECRET_KEY is not a Stripe test-mode key`**: only `sk_test_` and `rk_test_` keys are accepted, and none against production.
- **`seed: ... FAILED ON THE TARGET: ...`**, then `coverage_failures: count==0` fails: `setup()` found a journey broken on the target and switched it off; the summary's `coverage` says why for each, and the entries below cover the usual ones.
- **`the dummy account's probe reply was not the dummy engine's`**: the target does not route load-test accounts by address (`CHAT_LOAD_TEST_EMAIL_DOMAIN` unset, or another domain), so the probe ran on the real model; no other turn is sent and the run fails. Deploy the routing, or run with `LOAD_TEST_COPILOT_TURNS=off`.
- **`the dummy probe timeout`**, or turns that time out with no `finish`: the target's copilot executor is not running or not consuming the queue; the route answers 200 and heartbeats until the stream's TTL.
- **`the scripted probe ...: is the scripted model deployed ...`**: the scripted account's turn did not end with the script's final text, and the log says how it ended. A preview without the scripted model fails the turn at once; one whose scripted model lacks this folder's scripts answers it with a 400 that its journal and log explain (`no script named 'loadtest-turn'`); a reply that is the dummy's means the target runs every account on the dummy engine (`CHAT_TEST_MODE`). `copilot_turn_scripted` is off and the run fails; `LOAD_TEST_SCRIPTED_ACCOUNTS=0` runs without it.
- **`the scripted probe finished without the sandbox command's output`**: the model script ran, but `bash_exec` did not: the approval gate held the command or the sandbox failed. Scripted turns still run the engine, and the warning says only that.
- **`copilot_turn_scripted: skipped`** or **`copilot_turn_real: skipped`** with `no dummy account could probe the copilot executor first`: those turns are sent only after the dummy probe passes, and the run had no dummy account able to send it.
- **`seed: listing the expert templates failed`** or **`the target lists no expert templates`**: the expert journeys are off and the run fails; the target's roster has not been seeded from the skills catalog.
- **`seed: hiring the work expert for account N failed: HTTP 409 active_expert_limit`**: the account has 20 active experts; teardown archives the run's, so this is an account in `LOAD_TEST_ACCOUNTS` at its cap.
- **`expert_hire: account N has hired every template it may`**: a long or stress run has hired, on that account, every template but the work one; the account's later hire iterations do nothing.
- **`none of N seed runs completed`**: the executor is not running or not consuming its queue; the graph journey is off and the run fails.
- **`seed: listing store agents failed`** or **`seed: listing store skills failed`**: the store answered the seed with an error rather than a list (a skills `404` is the flag being off, a limit); a run that browses the store fails.
- **`ws_errors` fails**: graph runs' sockets did not open, were refused or were dropped; the log line for each says which (`graph run ... (poll): ...`). Check `K6_WS_URL` and the target's WebSocket server. The runs went on by polling, so `graph_run_ms` still has the executor's numbers.
- **`K6_BACKEND_URL ... on a production host sends load at production`**: a `K6_*_URL` in your shell points at production; unset it, or agree a production run with the team and set `K6_ALLOW_PROD=true`.
- **`dropped_iterations` fails**: k6 had no free VU when an iteration was due; the summary's `by_journey` says which journey. Each journey pre-allocates five times its expected concurrency (`configs/profiles.js`); a slower target needs more, so raise its `EXPECTED_ITERATION_SECONDS` there, or lower `RATE_SCALE`.
- **`429` during setup**: another sign-in from the same IP used up the burst. `setup()` waits and retries; it fails only after five refusals in a row.
- **`k6.sh: this command would use Grafana Cloud k6`**: run it on your machine instead.
- **`docker build` stalls on a credential helper**: `./k6.sh` builds with an empty temporary `DOCKER_CONFIG`; if your own `docker pull` hangs, do the same.
