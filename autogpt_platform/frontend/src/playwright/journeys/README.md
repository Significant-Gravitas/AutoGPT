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
    catches one, that's a finding. Report it, or mark the journey
    `test.fail()` with a link to the issue.
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
    A journey never shares a user with another one and never depends on CI's
    seeded accounts.
- **Make your own data.** Build or import the agents you need inside the
  journey. Name things uniquely (`Date.now()` suffix).
- **Assert invariants, not DOM trivia.** Use web-first assertions
  (`toBeVisible`, `expect.poll`); no fixed sleeps. When the UI and the API
  should agree, check both.
- **Keep each journey under ~60 s.** The test timeout is 90 s.
- **Reuse page objects** from `../pages/` and helpers from `../utils/`. Put
  helpers only journeys need (credits, users) in this folder.
- **Costs:** read them from the ledger (`credits.ts`), not by diffing the
  balance alone. Onboarding rewards (e.g. +100 for a first Library run) land
  mid-journey.

## Run locally

Start a full stack in the journeys configuration, then point Playwright at it.
Mirror CI's backend settings:

```bash
cd autogpt_platform
make init-env
python ../.github/scripts/hosted_journeys_env.py >> backend/.env
echo "CHAT_TEST_MODE=true" >> backend/.env
NEXT_PUBLIC_PW_TEST=true docker compose up -d --build
cd frontend
PLAYWRIGHT_JOURNEY_COST_PER_NODE=1 PLAYWRIGHT_JOURNEY_TIER=PRO \
  pnpm exec playwright test --config playwright.journeys.config.ts --project hosted
# one journey, headed:
pnpm exec playwright test --config playwright.journeys.config.ts --project hosted run-correctness --headed
```

The frontend must be built with `NEXT_PUBLIC_PW_TEST=true`; only that build
reads the hosted flag values the fixtures hand it.
`PLAYWRIGHT_JOURNEY_COST_PER_NODE` tells cost assertions what the stack bills
per node execution: `1` with `EXECUTION_COST_COUNT_THRESHOLD=1`, otherwise
leave it unset for `0`.

The tier grant uses `DATABASE_URL` from `frontend/.env`; if your stack's
Postgres isn't on `localhost:5432`, set `PLAYWRIGHT_JOURNEY_DATABASE_URL`.

### Projects: `hosted` and `selfhost`

`selfhost-journey.spec.ts` asserts that nothing hosted-only exists (no
billing, paywall or third-party hosts), so it fails by design on the hosted
configuration. The config runs it only in the `selfhost` project, against a
self-hosted build (see the spec's header); every other journey is in
`hosted`. CI's journeys leg runs `--project hosted`.

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
  config, so pass it `userAgent: JOURNEY_USER_AGENT` yourself.
- Users sign up as `journey-<id>@previews.agpt.co`, which the preview
  allowlist accepts. Override the domain with `PLAYWRIGHT_JOURNEY_EMAIL_DOMAIN`.
- Previews serve real PostHog flags, so `enable-platform-payment` is on there
  and a fresh user is paywalled: running an agent gets a 402. For journeys
  that run agents, sign in as a paid account instead with
  `PLAYWRIGHT_JOURNEY_EMAIL` and `PLAYWRIGHT_JOURNEY_PASSWORD`. Never put a
  real password in a log, commit or PR.
- Sign-ups are capped at 5 per client IP per 10 minutes, so a full run from
  one machine can hit `429`. Reuse an account as above, or run fewer
  journeys at once.

## In CI

Job `end-to-end tests (journeys)` in `.github/workflows/platform-fullstack-ci.yml`
is a matrix leg beside the smoke suite, on its own runner, so it adds no wall
clock. It's **advisory** for now (`ADVISORY_CHECK_RUNS` in
`.github/workflows/scripts/check_actions_status.py`): a failure shows on the
PR but doesn't block merging. It becomes blocking once it has proven stable.

- Raise `min_tests` for the journeys leg in the workflow matrix when you add
  tests; `.github/scripts/test_e2e_seed_cache_contract.py` pins it.
- Reports and traces: artifacts `playwright-report-journeys` and
  `playwright-test-results-journeys`.
- Check a new journey for flakiness before relying on it: dispatch the
  workflow on your branch with `journeys_repeat_each`, e.g.
  `gh workflow run platform-fullstack-ci.yml --ref <branch> -f journeys_repeat_each=10`.
- Keep the leg's Playwright step under ~200 s at 4 workers. Past that, shard
  it into more legs with `--shard=i/N` rather than lengthening it.

## Hosted vs self-hosted configuration

The journeys leg runs as close to the hosted product as one compose stack
allows. Flag values live in `hosted-flags.json`, read by both sides:

- **Backend:** `.github/scripts/hosted_journeys_env.py` turns them into
  `FORCE_FLAG_<NAME>` lines in `backend/.env`, plus `ENABLE_CREDIT=true`.
- **Frontend:** the fixtures write them to local storage before the app
  loads, and a `NEXT_PUBLIC_PW_TEST=true` build serves them
  (`services/feature-flags/pw-flag-overrides.ts`). That lets both legs share
  one cached frontend image; build-time `NEXT_PUBLIC_FORCE_FLAG_*` would mean
  a second, uncached Next.js build on every run.

Differences from hosted that are still open:

| Area               | Journeys leg                                                                                                                        | Hosted                                                                         |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Payments           | `enable-platform-payment` **on**; journey users get a paid tier without Stripe, so checkout, trials and Stripe webhooks go untested | On, with Stripe: NO_TIER users get a 402 until they subscribe or start a trial |
| Node execution fee | 1 credit per node (`EXECUTION_COST_COUNT_THRESHOLD=1`), so a run's cost is visible                                                  | 1 credit per 100 node executions                                               |
| Copilot            | `CHAT_TEST_MODE` dummy reply; no real turn pipeline, tools or cost                                                                  | Real models                                                                    |
| Flag source        | Fixed values from `hosted-flags.json`; JSON flags (model routing, tier limits) at code defaults                                     | PostHog/LaunchDarkly with targeting                                            |
| `BEHAVE_AS`        | `LOCAL` on both sides; the frontend's is build time                                                                                 | `CLOUD`                                                                        |
| Pods               | One `executor`, one `copilot_executor`                                                                                              | Many. Scaling needs the fixed host ports 8002/8008 removed first               |
| Frontend runtime   | Next.js standalone server in Docker                                                                                                 | Vercel (body size limits, edge routing)                                        |
| External services  | No Stripe, OAuth providers, Postmark or E2B; ClamAV, Redis, RabbitMQ and FalkorDB are real                                          | Real                                                                           |
| LLM keys           | None; blocks that need platform credentials fail                                                                                    | Platform keys                                                                  |
| Origin             | `http://localhost:3000`                                                                                                             | HTTPS                                                                          |
