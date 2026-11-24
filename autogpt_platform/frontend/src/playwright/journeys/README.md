# Journeys

A journey is a Playwright spec that walks one real user flow end to end and
asserts what must be true afterwards: the run reached COMPLETED, the Library
shows it, the balance moved by exactly the cost. Smoke specs
(`*-happy-path.spec.ts`) check that pages work; journeys check that the
product does what the user expects.

## Conventions

- **File name:** `src/playwright/journeys/<area>-journey.spec.ts`. Only files
  matching `*-journey.spec.ts` run, and only in the journeys config.
- **Import `test` and `expect` from `./fixtures`**, never from
  `@playwright/test` or `../coverage-fixture`. That gives every journey:
  - **The guard.** The test fails if the browser sees an API response of 400
    or more (`/api/...`), an uncaught page error, or a `console.error`. The
    failure lists each one with its URL, status and the start of the body.
    A journey that provokes an error on purpose declares it:
    ```ts
    test.use({
      expectedResponses: [{ status: 401, url: "/api/proxy/api/auth/user" }],
    });
    ```
    App-wide noise that is reviewed and harmless goes in `KNOWN_BENIGN` in
    `guard.ts`, with its reason. A real bug never goes there: if the guard
    catches one, that's a finding. Report it, and fix it or pin it (see
    **Known bugs**).
  - **Hosted flags.** The frontend serves the values in `hosted-flags.json`
    (see below). Change single flags per journey with
    `test.use({ flagOverrides: { "<flag>": value } })`; the rest keep their
    hosted values.
  - **`journeyUser`.** Ask for it to get a brand-new user who is signed in and
    has finished onboarding. With credits on, finishing onboarding grants the
    onboarding credits. With `PLAYWRIGHT_JOURNEY_TIER` set (CI: `PRO`), the
    user also gets that plan without Stripe (`subscription-tier.ts` writes the
    tier to the stack's database: a paid tier with no Stripe customer, the
    same shape as an admin grant), so the hosted paywall lets them run agents.
    A journey about buying a plan opts out with
    `test.use({ grantJourneyTier: false })`: its user starts with no plan,
    as a new hosted user does (`subscription-journey.spec.ts`).
    A journey never shares a user with another one and never depends on CI's
    seeded accounts.
- **More than one person.** Give each other person (an admin, a second user)
  a browser context of their own, made in a local `test.extend` fixture so it
  is closed after the test, with `serveHostedFlags` from `./fixtures` and its
  own `watchForUnexpectedErrors` guard, and sign them up with
  `provisionNewJourneyUser`. `grantAdminRole` (`admin-role.ts`) makes one an
  admin; it writes the stack's database, so skip with `adminGrantUnavailable()`
  where that can't be reached. `marketplace-journey.spec.ts` does all of this.
- **Undo what you change on the stack in a fixture's teardown,** not in a
  `finally` in the test's body: Playwright abandons the body when the test
  times out, but still tears its fixtures down. Wrap `provide()` in
  `try`/`finally`, as the admin fixtures in `marketplace-journey.spec.ts` and
  `route-crawl-journey.spec.ts` do; otherwise a long-lived stack keeps a
  journey user as a platform admin. Name the callback `provide`: ESLint's
  rules-of-hooks fails `use(...)` inside a named fixture.
- **Make your own data.** Build or import the agents you need inside the
  journey. Name things uniquely (`Date.now()` suffix).
- **Assert invariants, not DOM trivia.** Use web-first assertions
  (`toBeVisible`, `expect.poll`); no fixed sleeps. When the UI and the API
  should agree, check both.
- **Known bugs.** A journey that finds an open bug keeps running in CI,
  pinned to it:
  - File the issue and add it to the test's details:
    `{ annotation: { type: "issue", description: "<issue URL>" } }`.
  - Call `pinKnownFailure("#<n>: <what goes wrong>", "<check message>")`
    (`known-failure.ts`) right before the one check the bug breaks, and pass
    the message it returns to that check. Every check that must still pass
    goes above it.
  - CI accepts the test only if it fails once, on that check: it must
    report exactly one error, with that check's message
    (`.github/scripts/validate_playwright_json.py --allow-known-failures`). A
    cleanup in `finally`, the guard at teardown or a soft check failing as
    well fails the run, and so does the check starting to pass. When the fix
    lands, drop the pin.
  - One pin per test, and a one-line check message.
  - The check must fail as an assertion, not by timing out. Give it a timeout
    well inside the test's.
- **Keep each journey under ~60 s.** The test timeout is 90 s.
- **Reuse page objects** from `../pages/` and helpers from `../utils/`. Put
  helpers only journeys need (credits, users) in this folder.
- **Costs:** read them from the ledger (`credits.ts`), not by diffing the
  balance alone. Onboarding rewards (e.g. +100 for a first Library run) land
  mid-journey.

## The route crawl

`route-crawl-journey.spec.ts` opens every page in `src/app` as the person it
is for, each in a tab of its own watched by the guard on that tab alone
(`watchPageForUnexpectedErrors`), and fails with each page that didn't load
cleanly and why. A session's tabs share a browser context. Before checking a
page, the crawl waits, each wait capped, for its scripts to have run, its API
calls to go quiet and the browser to go idle. That is a heuristic for
"hydrated", not proof of it. A page loads cleanly when:

- the server answers it below 400;
- it ends up where it should;
- it shows content of its own;
- no error UI shows: the error card, the error page, or Next's not-found or
  exception page;
- every image under `public/`'s folders that it asks Next's image optimizer
  for exists there and is served (the crawl answers a missing one itself, so
  the server logs no error for it);
- none of its API requests stays open longer than 15 s;
- the guard sees no failed API call, page error or `console.error`.

The pages are read from `src/app` when the tests are listed, so a new page is
crawled as soon as it exists. It is crawled signed in, or as an admin under
`/admin`. Pages for anyone are crawled signed out too. Each page is opened on
its own: the crawl answers Next's link prefetches with nothing, so the pages
a page links to aren't rendered alongside it.

A crawl starts no visit too late for it, and the test's report, to end inside
the test's timeout (`test.info().timeout`). A page it runs out of time for is
annotated "unreached page", not failed: a slow runner is no broken page. A
page it opened and found broken still fails.

- **A page that sends the user on, needs query parameters, or is public**
  gets an entry in `ROUTE_RULES` (`route-table.ts`), with the reason. Say
  where it lands, what it shows, or which error response it gets by design.
- **A page only some stacks serve** is `localOnly` (only a local stack, such
  as the test-data page) or `hostedOnly` (only a stack that takes payments,
  such as Billing, which a self-hosted build without payments answers with
  the not-found page). Elsewhere the crawl annotates it "page not served
  here" and doesn't open it. A page that sends the user on to one is marked
  the same.
- **A page with a dynamic segment** (`[id]`) is crawled only with data a test
  samples (`route-samples.ts`). Until it has one, the crawl annotates it
  "unsampled route" and doesn't fail. Add the sample to one of the crawl's
  tests, and the route to `DYNAMIC_ROUTES` with that test (`sampledIn`). That
  test fails unless it crawls each route `DYNAMIC_ROUTES` gives it, or says
  why it can't on this stack.
- **A page with an open bug:** add it to `KNOWN_BUGS` (`route-table.ts`):
  the page, the issue, whom it shows for, what goes wrong, and a pattern
  matching the problems it causes. The page is then left out of its group
  and crawled in a test of its own per bug, pinned to it. Any other problem
  on the page still fails that test. A bug a plain visit shows only now and
  then, or a page with a dynamic segment, needs a visit that shows it every
  time, in the spec's `REPRODUCTIONS`; a hydration bug can open its page
  with the browser's clock ahead (`clockAheadMs`). A pinned test tries its
  visit up to three times, until the bug shows, so a busy runner can't make
  it pass; a try the crawl runs out of time for ends the tries, and with
  none made the test only notes the page. A bug other pages show too,
  depending on what they list, is `showsElsewhere`: their groups note it
  rather than fail on it.
- **When the fix lands,** drop the bug from `KNOWN_BUGS` and its visit from
  `REPRODUCTIONS`, and keep a visit that showed it every time as a test of
  its own that must load cleanly: the pages that show an age (#17309,
  #17310) still open with the browser's clock ahead.
- **A bug that shows only now and then**, with no visit that shows it every
  time, can't be pinned: CI accepts a pinned test only if it fails every
  run. Add it to `INTERMITTENT_BUGS` with its issue. The crawl then notes it
  on any page it shows on, with an annotation, rather than fail.

## Run locally

Start a full stack in the journeys configuration, then point Playwright at it.
Mirror CI's backend settings; `docker-compose.journeys.yml` adds the scripted
model AutoPilot chats with (see below):

```bash
cd autogpt_platform
make init-env
python ../.github/scripts/hosted_journeys_env.py >> backend/.env
NEXT_PUBLIC_PW_TEST=true docker compose -f docker-compose.yml \
  -f docker-compose.journeys.yml up -d --build
# The expert roster's workflows, then the roster again (see below):
docker compose -f docker-compose.yml -f docker-compose.journeys.yml run --rm \
  --no-deps rest_server python -m test.load_store_agents --local-media
docker compose -f docker-compose.yml -f docker-compose.journeys.yml up publish_skills
cd frontend
PLAYWRIGHT_JOURNEY_COST_PER_NODE=1 PLAYWRIGHT_JOURNEY_TIER=PRO \
  pnpm exec playwright test --config playwright.journeys.config.ts --project hosted
# one journey, headed:
pnpm exec playwright test --config playwright.journeys.config.ts --project hosted run-correctness --headed
```

Journeys that call the backend directly, such as a webhook delivery, use
`PLAYWRIGHT_BACKEND_URL` (default `http://localhost:8006`). Set it whenever
`PLAYWRIGHT_BASE_URL` points at a remote stack.

The expert roster's workflows are the official marketplace listings
(`backend/agents`), and `publish_skills` installs the roster when the stack
starts, skipping any workflow whose listing isn't there yet. CI loads the
listings before the stack starts; the commands above load them into the
running stack instead, then publish the roster again. `--local-media` gives the
listings and their creator placeholder images in the stack's own media storage,
as CI does, so no page fetches the production images the CSV points at.

The frontend must be built with `NEXT_PUBLIC_PW_TEST=true`; only that build
reads the hosted flag values the fixtures hand it.
`PLAYWRIGHT_JOURNEY_COST_PER_NODE` tells cost assertions what the stack bills
per node execution: `1` with `EXECUTION_COST_COUNT_THRESHOLD=1`, otherwise
leave it unset for `0`.

The tier grant and the copilot cost check use `DATABASE_URL` from
`frontend/.env`; if your stack's Postgres isn't on `localhost:5432`, set
`PLAYWRIGHT_JOURNEY_DATABASE_URL`.

### Chatting with AutoPilot: the scripted model

Journeys that chat run the real copilot pipeline (engine, tool calls,
persistence, streaming and reconnects, cost accounting) against
`autogpt_platform/scripted_model`, a fake model provider that plays each
model turn from a script. `docker-compose.journeys.yml` starts it and points
AutoPilot at it, in place of `CHAT_TEST_MODE`'s canned reply.

- **Write the script** in `model-scripts/<name>.json`: the text, tool calls
  (with their exact arguments) and token usage of each model turn, in order.
  The format is in `scripted_model/README.md`.
- **Pick it** with `scriptedRun("<name>")` from `scripted-model.ts`, and put
  its `marker` in the chat message.
- **Import the script** in the spec to know what to expect, as
  `copilot-build-agent-journey.spec.ts` does.
- **Check the run** with the helpers in `scripted-model.ts`:
  `expectEachTurnOnce` (every turn asked for once, every expected tool result
  seen, every reply delivered in full) and `expectChargedWhatWasBilled` (the
  copilot's cost ledger charged exactly what the model billed). They read the
  scripted model's journal at `localhost:4010`.
- **A block run with the user's own key:** the stack points Anthropic clients
  at the scripted model too, so an AI block run with an Anthropic key the
  journey's user saved lands there. `readJournalForKey` lists the calls made
  with that key, as `copilot-connect-card-journey.spec.ts` does. Its connect
  card steps (ask, add a key on the card, check the save) are in
  `connect-card.ts`.
- **An expert's kickoff** is a message the platform writes, so no marker can
  be typed into it. Put the marker in the expert's writing sample at the
  hire's voice pick: the system prompt carries the sample, and the scripted
  model looks there when no user message has a marker
  (`experts-journey.spec.ts`).
- **An id a tool returned**, such as a schedule's, passed on to the next
  tool call: `$from_tool_result` in the call's input reads it from that
  result (`scripted_model/README.md`).

- **Chat helpers** are in `copilot-chat.ts`. `recordChatStreams` records the
  body of every chat stream as the browser received it (Chromium keeps no
  streamed body for Playwright), and `cutChatStreams` fails the open ones the
  way a dropped connection does. `context.setOffline` can't do that: Chromium
  lets a stream that is already open finish.

A request the script can't answer (no marker, past the last turn, a tool the
platform didn't offer) fails the turn with the reason, which shows in the
scripted model's container log and journal. These journeys skip against a
deployed app, whose AutoPilot uses real models.

### Paying: the fake Stripe

The hosted configuration points the backend's Stripe at the scripted model
too (its "Fake Stripe" section), so a journey pays the way a user does: the
app opens the fake's checkout page, **Pay** sends the platform a signed
`checkout.session.completed` webhook and then returns to `success_url`.
`billing-journey.spec.ts` buys credits this way.

Plans work the same way. The fake sells Pro and Max at its own prices
(`scripted_model/scripted_model/fake_stripe/catalog.json`), and the hosted
configuration builds the platform's price flag from that file.
**Subscribe** starts the subscription and pays its first invoice, the billing
portal's **Cancel plan** sets it to cancel at the period's end, and the
journey ends a period with `endSubscriptionPeriod`.
`subscription-journey.spec.ts` buys Pro, cancels or resumes it, and lets the
period end. The helpers are in `fake-stripe.ts`:

- `checkoutSessionID` reads the session from the checkout page's URL.
- `holdWebhooks` keeps the payment's webhook until `deliverEvent` sends it,
  for a webhook that arrives after the user is back.
- `stripeEvents` lists the events about a session with each delivery's HTTP
  status from the platform; `deliverEvent` sends one again, as Stripe
  retries, and `sendEvent` sends a new event type about it.
- `unservedStripeRequests` lists Stripe calls the fake doesn't model since a
  `stripeRequestCursor` the journey took when it started; a journey that buys
  something checks it's empty.
- `customerStripeEvents` lists every event about a customer's objects, to
  check the platform acknowledged each one; an event the platform's own API
  call raised arrives a moment after that call's answer.
- `stripeObject` reads an object as the fake holds it (a subscription's
  period, an invoice's number), and `endSubscriptionPeriod` ends a
  subscription's period now: it renews, or ends if it was set to cancel.

They skip against a deployed app, whose Stripe is real (test mode on
previews).

### Projects: `hosted` and `selfhost`

`selfhost-journey.spec.ts` asserts that nothing hosted-only exists (no
billing, paywall or third-party hosts), so it fails by design on the hosted
configuration. The config runs it only in the `selfhost` project, against a
self-hosted build (see the spec's header); every other journey is in
`hosted`. CI's journeys legs run `--project hosted`.

**TODO:** a `journeys-selfhost` CI leg (compose stack with hosted flags off,
no `CHAT_TEST_MODE`, since the spec expects "No AI connection") would run it
on every PR. It's not added yet: it's a third runner per run, and its config
needs settling first.

## Run against a preview (`!deploy`)

```bash
PLAYWRIGHT_BASE_URL=https://autogpt-pr-<N>.vercel.app \
  pnpm exec playwright test --config playwright.journeys.config.ts --project hosted
```

- Journeys present a normal Chrome user agent (`user-agent.ts`; override with
  `PLAYWRIGHT_USER_AGENT`). PostHog's edge returns no flags at all to a
  `HeadlessChrome` user agent, so with the default headless one, dev and
  previews would serve every flag off: no paywall, no experts, old layout.
  A context you create with `browser.newContext()` doesn't inherit the
  config, so pass it `userAgent: JOURNEY_USER_AGENT` yourself, and call
  `markChangelogSeen(context)` so its pages get the changelog from the
  fixture instead of agpt.co.
- Users sign up as `journey-<id>@previews.agpt.co`, which the preview
  allowlist accepts. Override the domain with `PLAYWRIGHT_JOURNEY_EMAIL_DOMAIN`.
- Previews run with `ENABLE_PLATFORM_PAYMENT` on, so a fresh user is
  paywalled: running an agent gets a 402. For journeys
  that run agents, sign in as a paid account instead with
  `PLAYWRIGHT_JOURNEY_EMAIL` and `PLAYWRIGHT_JOURNEY_PASSWORD`. Never put a
  real password in a log, commit or PR.
- Sign-ups are capped at 5 per client IP per 10 minutes, so a full run from
  one machine can hit `429`. Reuse an account as above, or run fewer
  journeys at once.
- Setting `PLAYWRIGHT_JOURNEY_DATABASE_URL` for a deployed stack also lets the
  marketplace journey make a temporary admin there and approve its test
  listings, which stay on that stack. The role is taken back at teardown, but
  tokens already issued keep it until they expire, within an hour
  (`admin-role.ts`). Without the variable, that journey skips.

## In CI

Journeys run as two Playwright shards, jobs `end-to-end tests (journeys 1/2)`
and `end-to-end tests (journeys 2/2)` in
`.github/workflows/platform-fullstack-ci.yml`. Each is a matrix leg beside the
smoke suite with its own runner and stack, so they add no wall clock. They're
**advisory** for now (`ADVISORY_CHECK_RUN_PREFIXES` in
`.github/workflows/scripts/check_actions_status.py`): a failure shows on the
PR but doesn't block merging. They become blocking once they have proven
stable.

- **Shards:** the two legs are identical except `shard_index`; Playwright
  gets `--shard=<shard_index>/<shard_total>`. The journeys config inherits
  `fullyParallel: true` from `playwright.config.ts`, so Playwright cuts the
  listed tests, not files, in file order, into slices of equal count: shard 1
  runs the first half of the list. The slices hold equal counts, not equal
  times: the long journeys
  aren't spread evenly through the list.
- **The floor:** `min_tests` on the journeys legs is the pinned inventory of
  the whole suite, the same number on every leg, not a per-shard count. Raise
  it when you add journeys (`.github/scripts/test_e2e_seed_cache_contract.py`
  pins it too). Count them with
  `pnpm exec playwright test --config playwright.journeys.config.ts --project hosted --list`;
  the last line reads `Total: N tests in M files`, and listing needs no
  stack. Each leg runs that listing while its stack builds, in "Count the
  shard's tests", and fails if the total is below `min_tests`. It then lists
  its own shard, with the run's `--shard` and `--repeat-each`, and
  `validate_playwright_json.py` gets that count as `--min-tests`, so a shard
  that runs fewer tests than it listed fails too.
- **Reports and traces:** each shard uploads its own pair of artifacts,
  `playwright-report-journeys-<shard_index>` and
  `playwright-test-results-journeys-<shard_index>`. A test's trace and
  `results.json` entry are in the artifact of the shard that ran it; download
  both with
  `gh run download <run> -p 'playwright-test-results-journeys-*'`.
- Check a new journey for flakiness before relying on it: dispatch the
  workflow on your branch with `journeys_repeat_each`, e.g.
  `gh workflow run platform-fullstack-ci.yml --ref <branch> -f journeys_repeat_each=10`.
  The repeated list is split the same way, so each journey still runs 10
  times across the shards.
- **When to add a shard:** keep each shard's Playwright step under ~200 s at
  4 workers. When the slower one goes past that, add a shard rather than
  lengthening it: copy a journeys leg with the next `shard_index`, and set
  `shard_total` on every leg to the new count. Every leg's name changes
  (`journeys 1/2` becomes `journeys 1/3`); the status check treats any
  `end-to-end tests (journeys …` check as advisory, so it needs no edit. Then
  update `.github/scripts/test_e2e_seed_cache_contract.py`, which pins the
  shard legs and their arguments (`test_journeys_run_as_identical_shards`)
  and their job and artifact names
  (`test_each_leg_has_its_own_job_and_artifact_names`). Each shard costs a runner that builds its own stack, about
  10 minutes before the first test.

## Hosted vs self-hosted configuration

The journeys legs run as close to the hosted product as one compose stack
allows. Flag values live in `hosted-flags.json`, read by both sides:

- **Backend:** `.github/scripts/hosted_journeys_env.py` turns them into
  `FORCE_FLAG_<NAME>` lines in `backend/.env`, plus `ENABLE_CREDIT=true`
  and `ENABLE_PLATFORM_PAYMENT=true`. It also sets the one JSON flag the
  stack needs, which only the backend reads: `copilot-tier-stripe-prices`,
  built from the fake Stripe's catalogue. `FORCE_FLAG_<NAME>` holding a JSON
  object or array sets a JSON-valued flag (`_NON_BOOLEAN_FLAG_VALUES` in
  `backend/util/feature_flag/__init__.py`), and the backend logs a warning
  naming the variable the first time it reads one.
  `NEXT_PUBLIC_FORCE_FLAG_<NAME>` never sets a JSON-valued flag.
- **Frontend:** the fixtures write them to local storage before the app
  loads, and a `NEXT_PUBLIC_PW_TEST=true` build serves them
  (`services/feature-flags/pw-flag-overrides.ts`). That lets both legs share
  one cached frontend image; build-time `NEXT_PUBLIC_FORCE_FLAG_*` would mean
  a second, uncached Next.js build on every run.

Differences from hosted that are still open:

| Area               | Journeys leg                                                                                                                                                                                                                              | Hosted                                                                         |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Payments           | `ENABLE_PLATFORM_PAYMENT` **on**; most journey users get a paid tier without Stripe. Plans and credit top-ups go through the fake Stripe's checkout, billing portal and webhooks; trials and paid plan changes (Pro to Max) stay untested | On, with Stripe: NO_TIER users get a 402 until they subscribe or start a trial |
| Node execution fee | 1 credit per node (`EXECUTION_COST_COUNT_THRESHOLD=1`), so a run's cost is visible                                                                                                                                                        | 1 credit per 100 node executions                                               |
| Copilot            | The scripted model: the real engine (Claude Agent SDK), tools, persistence, streaming and cost, with scripted model turns. Memory (Graphiti) calls fail, so turns run without memory                                                      | Real models                                                                    |
| Flag source        | Fixed values from `hosted-flags.json`; plan prices from the fake Stripe's catalogue; other JSON flags (model routing, tier limits) at code defaults                                                                                       | PostHog with targeting                                                         |
| `BEHAVE_AS`        | `LOCAL` on both sides; the frontend's is build time                                                                                                                                                                                       | `CLOUD`                                                                        |
| Pods               | One `executor`, one `copilot_executor`                                                                                                                                                                                                    | Many                                                                           |
| Frontend runtime   | Next.js standalone server in Docker                                                                                                                                                                                                       | Vercel (body size limits, edge routing)                                        |
| External services  | No OAuth providers, Postmark or E2B; Stripe is the fake in the scripted model; ClamAV, Redis, RabbitMQ and FalkorDB are real                                                                                                              | Real                                                                           |
| LLM keys           | None; blocks that need platform credentials fail. Anthropic clients point at the scripted model, so a block run with the user's own Anthropic key reaches it                                                                              | Platform keys                                                                  |
| Origin             | `http://localhost:3000`                                                                                                                                                                                                                   | HTTPS                                                                          |
