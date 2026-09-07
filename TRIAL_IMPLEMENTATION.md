# Card-required trials

## Scope and decisions

Build from `origin/dev` at `557a9f54301359e2ef506688556a8da46be0417f` in
`feat/card-required-trials`. The goal includes Stripe checkout, a distinct trial
tier, configurable usage limits, trial emails, frontend enrollment and billing,
analytics, staging validation, and an explicitly verified live rollout.

The user confirmed that pricing and allowances await the analytics PRs. Duration,
destination tier/price, and trial allowances must be tunable without code changes.
Initially eligible: new users. Former free beta users may be included later through
an explicit server-controlled cohort. A valid card on file is required.

The user clarified that the one-time credit benefit is the existing onboarding
grant, not an additional trial grant. Current `ONBOARDING_COMPLETE` awards 300
credits ($3), keyed `REWARD-{user_id}-ONBOARDING_COMPLETE`. Trial enrollment must
reuse that identity, never top up users who already received it, and allow the
amount to be tuned for future first-time recipients. Other earned onboarding
rewards are separate and unchanged.

Proposed policy: one trial per user; exclude previous paid subscribers and previous
trial users, including canceled trials. Automatically convert at the disclosed
price after the disclosed duration unless canceled. Canceling preserves access
until the disclosed trial end and prevents conversion. Card setup must complete
before granting trial access. A successful setup does not guarantee the future
charge will succeed.

## Implementation requirements

- A server-side offer config defaults to unavailable and validates the entire
  offer, including a version, signup cutoff, duration, paid tier, billing interval,
  daily/weekly/total usage allowances, and optional beta cohort eligibility.
- Record the accepted offer, Stripe price/amount/currency, user and subscription
  identifiers durably. Changing config affects future offers only. Never trust
  client-provided trial terms, signup dates, or eligibility.
- Serialize enrollment and use Stripe idempotency. Retrying or opening multiple
  tabs must not create multiple chargeable subscriptions. Abandoning Checkout
  must not consume a trial; completing it must consume eligibility permanently.
- Require card-only Checkout and off-session setup. Verify current Stripe state
  on fulfillment; a webhook payload or browser success URL is not card proof.
- Model `TRIAL` independently of the paid destination tier. Webhook handling,
  reconciliation, cached user state, and entitlement checks must agree. Expiration
  must deny access even if a webhook is delayed. Paid access requires the first
  conversion payment to succeed; failed payment and authentication-required paths
  must remain recoverable.
- Configure trial limits independently of paid tiers. Audit concurrent runs,
  background work, BYOK, daily resets, and counter failure. Existing paid-plan
  pre-turn cost checks are approximate, not a strict reservation guarantee.
- Send trial welcome, upcoming conversion reminder, cancellation confirmation,
  expiry, successful conversion, and payment-failure emails with actual accepted
  price/date and billing/cancellation links. Cover webhook retries, out-of-order
  delivery, queue failure, and email rendering.
- Display trial duration, card requirement, conversion price and timing, allowance,
  and cancellation action in onboarding and billing. Regenerate the API client.
- Preserve DataFast attribution. Emit trial-specific PostHog events with the
  accepted offer version and durable user identity; never count a zero-dollar
  trial as paid conversion.
- Test real Stripe test-mode Checkout and test-clock transitions, staging
  integration and all relevant local/CI checks. Configure the necessary webhook
  events and reminder settings. Verify rollout separately from code readiness.

## Current evidence

- New worktree created clean; platform/backend/frontend/database `.env` files
  match their source without exposing values.
- GitHub CLI works outside the network sandbox.
- Stripe CLI `--project-name default` resolves AutoGPT account
  `acct_1OD9KOEVwK4k8ivI`; the ambient profile resolves the unrelated PR Reviewer
  sandbox. Always specify the AutoGPT profile.
- Test-mode dev webhook `we_1QoQSPEVwK4k8ivIMOyLjLIH` points to
  `https://dev-server.agpt.co/api/credits/stripe_webhook`, uses API `2023-10-16`,
  and does not currently subscribe to `customer.subscription.trial_will_end`.
- At the fetched base, `credit.py` and periodic reconciliation map `trialing` directly
  to the paid price's tier. `notifications/lifecycle.py` explicitly has no trial
  handler and Checkout currently sends the paid welcome.
- Analytics PR #14290 at `1c9c1f496421aa12ad3ca2d86445f727536a4083`
  provides unit economics and experiment views; PR #14287 at
  `c726c35b227eeac8a1f584e1667c7c6ebce07f2a` provides first-write-wins
  experiment assignments. They are inspected dependencies, not merged evidence.
- A later read-only status refresh found both still open: #14290 head
  `b3700b0e1e923540faee82f0fcca86587508ba52` (BEHIND), and #14287 head
  `d302ac9e9b96f42a7fc6df49c15c004dba35851d` (BLOCKED). Those newer diffs have
  not yet been reviewed for trial integration.
- Backend and frontend dependencies are installed. Prisma generation needed
  Node 24 and the worktree Poetry virtualenv first on PATH due to a stale global
  launcher. The frontend client was regenerated from the worktree OpenAPI spec.

## Implementation and validation snapshot (2026-09-04)

The implementation is on `feat/card-required-trials` and is not deployed.
Implemented: trial schema/config,
accepted terms and onboarding-grant reuse, card-required Checkout, current-state
Stripe reconciliation, dedicated trial limits, lifecycle email templates and
dispatch, onboarding/billing UI, and trial-specific analytics.

- A shared PostgreSQL advisory transaction lock now guards both paid and trial
  Checkout creation. Competing subscription sessions must expire successfully
  before creating or resuming a checkout. Errors propagate rather than allowing
  another potentially chargeable session. Paid checkout checks current Stripe
  trial subscriptions even when the local tier has not yet caught up.
- The in-place paid-plan modification path also checks current Stripe trial
  metadata/status before making any changes, rather than relying only on the
  possibly stale local tier. Its stale-tier bypass was reproduced test-first.
- Pending-offer GET and POST both respect disabled enrollment. This does not
  expire already-issued Stripe links or revoke accepted trial access.
- A disabled pending checkout no longer renders as an ended trial. Zero-credit
  offers do not advertise an onboarding credit grant.
- Stripe live portal configuration `bpc_1OLdyEEVwK4k8ivILQ6fLO63` currently has
  subscription updates disabled, card updates enabled, and cancellation at period
  end enabled. This is read-only evidence, not a configuration change.
- Whole-backend `poetry run format` (including pyright) passed.
- Feature checkpoint `988c69b591ec1a4a3e7084e0c313608c7da41bd0` passed all
  configured commit hooks. The trial fixture's opaque test IDs were replaced
  with stable labels without suppressing scanner rules. A separate all-files
  secret scan still reports pre-existing findings outside this change; it is
  not a passing repository-wide security audit. Its only baseline edit was an
  existing fingerprint's line number (12 to 13) and the generated timestamp.
- The standalone shared-library commit hook exposed four unresolved existing
  lazy backend imports. Added the sibling backend source path to its pyright
  config; standalone type checking and all **247 shared-library tests passed**.
- Focused backend suite: **397 passed**, including offer validation, card and
  invoice state tests, existing billing/rate-limit regressions, notification
  rendering, checkout failure handling, and API contract/ownership assertions.
  These tests mock external services and use `--noconftest`; they do not prove
  hosted Checkout or live delivery.
- Disposable PostgreSQL integration suite: **8 passed**. Covers first-write-wins
  enrollment, atomic recorded costs, existing-grant deduplication, lock release,
  and both paid-first/trial-first concurrent checkout entrypoints.
- Initial full frontend run: 6,104 passed / 25 failed. Fifteen failures came from
  an onboarding test's missing PostHog mock export (fixed). Rerunning all failing
  files plus the trial tests with two workers: **89 passed across 9 files**, with
  no expected failures remaining. The complete required frontend sequence then
  passed in order: format, lint, types, and `pnpm test:unit --maxWorkers=2`:
  **6,131 passed across 574 files**. Existing warnings remain.

## Required remaining audits before launch

- Real Stripe hosted Checkout: successful card setup, declines, 3DS, abandonment,
  and resuming setup when a subscription exists but no verified card is recorded.
- Check the platform-payment flag alongside trial enrollment, and verify card
  updates through the portal, including customer versus subscription default
  payment methods and payment-recovery behavior.
- Stripe test-clock conversion success/failure/cancellation; current price and
  quantity must agree with accepted terms before first conversion. Portal settings
  alone are not sufficient protection against other subscription-update paths.
- Crash/timeout recovery across paid and trial Checkout, including a lost Stripe
  response and transactions expiring while a remote creation remains in flight.
  A database lock alone is not a remote-operation fencing guarantee.
- Total spend currently uses atomic recorded costs plus pre-turn checks, not
  strict reservations. The user accepted this soft-cap behavior for phase one
  on 2026-09-06, including overshoot from concurrent/in-flight work. A strict
  dollar ceiling is not a launch requirement. Continue validating attribution,
  background work, BYOK, and failure behavior; do not advertise a hard ceiling
  or a verified maximum overshoot.
- Notification reliability: validate the durable recovery path through the real
  broker/RPC/provider, removed/restored-card notices, and visual email QA.
- Expand API/frontend cancellation, confirmation, auth-switch and trial-budget
  tests; Storybook and desktop/mobile rendered UI validation.
- Revalidate full backend CI and exact-head PR review. Migration and local check
  results are recorded below; the PR remains draft pending the launch gates.
- Choose final offer values from analytics; configure trial flag, webhook reminder
  event subscription, staging, and the approved live rollout. Nothing is live yet.

## Actual Stripe test-mode simulation

Used the verified AutoGPT account and API `2025-02-24.acacia` (the installed SDK's
version). Created only test-mode objects with no customer email addresses and a
`validation_run=card-trial-20260904` marker; no real app user IDs or live charges.
This exercised the subscription API directly, not hosted Checkout or our webhook
delivery. The $50/month price is a test fixture, not a proposed launch offer.

Clock `clock_1UC4sdEVwK4k8ivIMgiaQs8r` advanced from `1788523200` to `1789135200`
and returned `ready`. The three seven-day trials ended at `1789128000`:

| Scenario | Test customer | Test subscription | Result |
| --- | --- | --- | --- |
| Success | `cus_VCTrrploZaO5mt` | `sub_1UC4uNEVwK4k8ivI3xLlHyHV` | active; first cycle invoice paid 5000 cents |
| Decline | `cus_VCTrlbtDVItW01` | `sub_1UC4uNEVwK4k8ivIusaIpjin` | past_due; first cycle invoice open, paid 0 |
| Cancellation | `cus_VCTrLS6lPAXi9n` | `sub_1UC4uMEVwK4k8ivIakKzjEdb` | canceled; latest invoice remains the original $0 invoice |

All three initial invoices were `paid` with `subscription_create` and amount 0.
Captured safe response fields are in
`backend/backend/data/test_data/trial_stripe_simulation.json`; replay tests feed
them into the actual entitlement decision; all **6 captured-state tests pass**.
Fixture object IDs use deterministic test labels; response states, dates, card
expiry, and monetary amounts are unchanged. Actual resource IDs are recorded above.
The test clock/customers/subscriptions
still exist for follow-up validation. No live Stripe configuration was changed.

## Hosted Checkout and fulfillment follow-up

- Verified the worktree's configured Stripe key is test-only and reaches AutoGPT.
  A temporary guarded harness uses the real checkout/confirmation code and the
  disposable database, injecting only the trial offer and test price selection.
- Test customer `cus_VCUA4PuWBWfp1O` maps only to local test user
  `d19b832a-a954-575a-84c9-9c6e3d420384`, enrollment
  `1b705787-0bd4-4553-84f9-2246ff872f3f`. No real platform user was created.
- Retrying reused the original open Checkout. Explicitly expiring that owned test
  session and retrying produced a new session while preserving the accepted
  offer, NO_TIER, no verified card, and unconsumed eligibility. Latest session:
  `cs_test_a11vpp7nbCpXOVEpcKEYixQng4VUVWG3dQIqTCN4zCinyvuWc4FSLCgUH2`.
  The old test session is expired and cannot be completed.
- Hosted page interaction remains unverified. Cua's automatic isolated-profile
  discovery found no eligible browser. Auto-review rejected using the existing
  Chrome PID for isolated preparation due to authenticated-profile exposure risk.
  That request was not retried or bypassed. User approval was requested for native
  controls in the new empty browser window without debugging or other-tab access.
- New guards respect the platform-payment flag, require matching completed
  card-only Checkout before consuming/granting a trial, preserve consumption when
  a card is removed after completion, and reject a mismatched Stripe price or
  quantity before first conversion. Stale earlier-attempt events do not rewrite
  current enrollment state. Unfinished own checkouts are distinguished from prior
  subscription history. Focused backend suite now **418 passed**; whole-backend
  format and type checks also passed.
- Billing auth-isolation PR #14226 is still open at
  `5de13ac07cd3f9759a2ab02d3b0fe79f965f6056` (BLOCKED). General payment/invoice
  components need that account-switching protection validated before launch.
- Fresh origin/dev is `a242bc9392a8a062cce48a7913b57a833b3b909b`. It includes
  shared async Stripe calls/timeouts (#14324), Redis 8.1 (#14334), and DataFast
  onboarding/billing instrumentation (#14186/#14187); integration is next.
- Conversion now stores `stripeConversionInvoiceId` alongside `convertedAt`.
  Subsequent reconciliation preserves both, and only that invoice can drive the
  trial-converted notice. Two test-first regressions reproduced the old behavior;
  first-invoice/renewal tests and a real database round-trip now pass. This does
  not make the notification queue durable: replay of the same invoice after the
  Redis TTL and the claim-before-publish crash window motivated the outbox
  implementation recorded below.

## Dev integration validation (2026-09-04)

- Merged dev at `a242bc9392a8a062cce48a7913b57a833b3b909b` without rebasing;
  kept the fail-closed competing-checkout expiration behavior
  and the newly merged DataFast onboarding step tracking.
- All trial Stripe requests now use the shared async timeout/metrics helper.
  Added bounded pagination so later pages cannot bypass the timeout, and tests
  for pagination, an incomplete empty page, and timeout during subscription-history
  inspection. These checks fail closed before a second checkout is created.
- Regenerated the client from the merged backend OpenAPI schema, resolving the
  missing admin impersonation endpoint types. The frontend format/lint/types
  sequence passed. Its first full test run had 6,150 passing and one stale toast
  failure in NeedsAttentionList. Fixed test isolation by awaiting the previous
  mutations and dismissing singleton toasts; all nine tests in that file passed.
  The repeated complete frontend sequence passed in order: format, lint, types,
  and **6,151 tests across 578 files**, in 409.78 seconds. No expected failures.
- Updated existing billing tests for the new trial database boundary and Stripe
  pagination interface. Replaced the obsolete test asserting trial-ending events
  are ignored with a reconciliation-and-reminder dispatch assertion.
- Expanded isolated backend suite: **557 passed**, including existing billing,
  all copilot rate-limit tests, and new conversion identity tests. Two database
  metadata tests were explicitly deselected; five legacy refund integration
  tests require the full server fixture and were not rerun in this isolated suite.
- Disposable trial PostgreSQL integration: **9 passed**. Shared-library suite:
  **247 passed** after the Redis dependency update. Whole-backend formatting and
  type checking passed. Whole-schema Prisma formatting also fixed existing
  alignment; retain those formatter changes per repository instructions.
- Applied the complete revised migration to a fresh pre-trial schema in local
  database `trial_migration_1788564518918`; both enums, the table, five indexes,
  and the foreign key were created. The separate local `trial_test` database was
  updated additively for the new field; no production database was touched.
- Stripe's installed Checkout SDK accepts `trial_period_days >= 1`; the separate
  absolute `trial_end` parameter requires at least 48 hours. This implementation
  uses days, so the configurable minimum of one day matches that API contract.
- Stripe's [customer email settings](https://docs.stripe.com/billing/revenue-recovery/customer-emails)
  offer a seven-day trial reminder and payment confirmation/recovery links.
  Verify reminder settings and the cancellation link before launch; do not assume
  the three-day `trial_will_end` webhook alone covers card-network requirements.
- Billing-auth PR #14226 and analytics PRs #14290/#14287 remain open at the heads
  recorded above. There are no Sentry connector tools available in this session;
  runtime validation still needs the deployed environment, not just local tests.
- Spend audit found and fixed a specific mismatch: the trial-aware remaining
  budget helper returns the actual remaining amount (including zero), but
  `copilot/sdk/service.py::_resolve_dynamic_max_budget_usd` applied the paid
  $0.50 floor. Trial SDK budget resolution now preserves five cents as five cents
  and raises before dispatch for zero, negative, or non-finite remaining budget.
  Five test-first failures reproduced the bug; all **12 trial/paid budget resolver
  tests pass**. Actual provider enforcement and the user-visible zero-budget
  recovery path still need end-to-end validation. The existing active-turn admission is
  also explicitly non-locked; it is not a spend reservation. Missing provider
  costs bypass counters, and late settlement currently depends on the user's
  current tier/status rather than the enrollment funding the original turn.

## Outstanding launch choices

Draft [PR #14353](https://github.com/Significant-Gravitas/AutoGPT/pull/14353) was opened
against dev at `057bbc21d6e5d455c04851e7b15cfd47105ad82e`. That merge commit passed
its configured hooks; both secret scanners also passed against the complete trial
diff from the merged dev ref. Initial CI ran at that head; failures and their
follow-up fixes are recorded below.
The worktree was clean after that push. No merge or production activation occurred.

The final offer amounts/duration/price and launch cutoff remain unset pending
unit-economics data. The initial trial config must remain off until these values
are chosen and the complete integration is verified. Onboarding credits are a
separate wallet from the operator-side chat spend budget.

## Trial email recovery checkpoint

Implementation checkpoint: `a415e380c3`. All configured commit hooks passed.

- Added a trial-only PostgreSQL outbox with immutable payloads, semantic unique
  keys, owner-fenced five-minute leases, bounded retries, and retained terminal
  outcomes. The new `trial_notifications_v1` queue carries only the durable ID.
  A registered one-minute scheduler pass re-publishes due or abandoned deliveries;
  the existing NotificationManager runs the consumer through DatabaseManager RPC.
- The producer records the notice before publishing, acknowledges a durable
  intent even if RabbitMQ is unavailable, and emits analytics only on first
  creation. API and webhook cancellation notices share a persisted revision;
  repeated cancel/resume cycles receive distinct identities.
- A trial-only async Postmark sender requires configured credentials, preserves
  both HTML/plain-text bodies and service-mail preference links, and attaches
  `trial_notice_id`. Retried delivery searches Postmark for prior acceptance
  before sending. Missing configuration, provider lookup failures, and unverified
  recipients are retryable instead of being treated as a successful send.
- Delivery work has a 240-second deadline inside the 300-second lease. This is
  bounded at-least-once delivery, not an exactly-once guarantee: provider indexing
  lag after an ambiguous send can still allow a duplicate. Exhausted attempts
  remain visible as failed rows and are logged for operations.
- Postmark authentication and the metadata-search API were verified with a
  read-only lookup for a random nonexistent notice ID. No email was sent. Mocked
  HTTP tests verify the actual Postmark request/response shape and failure paths.
- Database tests caught raw JSON parsing and immediate-claim timing issues; both
  are fixed. Initial intents are immediately due, while leases and retry/wake
  scheduling use the database clock. A separate test exposed Prisma's emulated
  enrollment-upsert race; enrollment now inserts under the unique user key and
  reads the winning row after a conflict.
- All **16 trial/outbox database integration tests passed**. They are now opted
  into the isolated GitHub CI database, with exact local/CI target guards and
  shared-connection ownership preserved. Both migrations applied from a fresh
  pre-trial schema in `trial_outbox_migration_1788568049868`.
- Expanded final isolated regression suite: **727 passed**, including all
  notification tests, trial API/fulfillment, existing billing, rate limits, and
  trial SDK budgets. Whole-backend formatting and type checks passed. The
  frontend code and public API schema were not changed by this checkpoint.
- CI at `057bbc21d6` exposed four Python 3.11 test failures (13,854 passed): two
  tests relied on a local return-URL setting, and the shared service-email fixture
  omitted TRIAL_UPDATE. Adding that fixture also caught a missing plain-text
  preferences link. All are fixed locally. Patch-coverage gates were also red;
  they remain launch gates, not checks to disable.
- Remaining email work: recording/recovering notices when a process dies
  between the subscription-state commit and notice creation; missed reminders;
  stale checkout-attempt handling; real broker/RPC/worker/provider validation;
  failed-delivery monitoring and recovery operations; and visual email QA.
  The outbox is not yet a completed end-to-end email cutover or a live rollout.

## Missing-notice and stale-delivery recovery checkpoint

- Added a database scan for consumed trials missing their expected durable
  notices, including upcoming three-day reminders without a webhook and expired
  local trial snapshots needing a Stripe refresh. It uses keyset pagination;
  one failing enrollment cannot starve later pages. Existing failed deliveries
  stay failed and visible instead of being recreated with a new identity.
- The existing one-minute notification recovery pass now refreshes each selected
  subscription through DatabaseManager RPC, reloads the saved trial, and creates
  the notices appropriate to its current state. Existing due deliveries are
  still published when the missing-intent scan raises. Queue outages leave new
  intents durable. Recovery is eventual; it does not send obsolete intermediate
  state changes after the subscription has moved on.
- Every new trial payload carries its semantic notice key. Before sending, the
  worker refreshes Stripe and database state and checks the revision, accepted
  terms, end date, subscription ownership, and checkout attempt. Old cancellation
  revisions, changed dates, and removed cards suppress stale mail. Earlier owned
  checkout attempts are acknowledged without another notice. Unidentified legacy
  payloads fail closed; no trial feature version has been deployed yet.
- Test-first checks reproduced 11 missing-notice database cases, 11 worker/policy
  cases, and four stale-notice cases before implementation. All expected-failure
  markers were removed. Final local verification: **445 notification/trial/billing
  tests**, **177 trial API/rate-limit/SDK-budget tests**, and **29 disposable
  PostgreSQL tests passed**. Whole-backend format, generated Prisma stubs, pyright,
  and `git diff --check` passed. Existing test/dependency warnings remain.
- The database suite includes actual missing welcome/reminder reconstruction and
  immutable deduplication during a mocked queue outage. Stripe, broker transport,
  and provider delivery in that test remain mocked; no real emails were sent.
- CI at the preceding head `621e13a5f383d0cd4bda932d55373f7449b271cb` finished
  with all backend Python versions, frontend tests, E2E, and image checks passing.
  Backend patch coverage reached **83.38% / 80% required**. Frontend patch coverage
  remained **46.15% / 70% required**, so the aggregate PR-status gate was red.
  These results do not cover this newer checkpoint until CI reruns.
- Still required: real broker/RPC/provider and rendered email validation; failed
  delivery monitoring/recovery; audit reactivation of previously suppressed
  notices after card restoration; hosted Checkout/card fallback and payment
  recovery; concurrency/late-settlement spend guarantees; frontend coverage and
  account-switching/UI QA; final offer choices, Stripe configuration, staging,
  independent review, and the approved live rollout. Trial enrollment remains off.

## Frontend confirmation and account-isolation checkpoint

- Added 17 UI-driven integration cases using generated MSW handlers: cancellation
  success/failure/retry, pending-button behavior, accepted checkout token and return
  destination, expired-offer refresh, status-query retry, yearly/zero-decimal offer
  display, expiry, account switching, and backend confirmation/confirmation retry.
- Tests reproduced an inactive HTTP 200 response silently finishing confirmation,
  and an older GET overwriting a successful confirmation with an obsolete trial
  offer. Confirmation now shows a retryable inactive-trial error. A shared,
  user-scoped cache update cancels older status requests before publishing either
  confirmation or cancellation and rechecks account identity across awaits.
- Followed the React guidance on keeping mutation work in its interaction path
  and deriving account-specific visible state instead of copying it into effects.
  Existing grant wording and tunable offer values are unchanged.
- Complete required sequence passed: whole-frontend `pnpm format`, `pnpm lint`,
  `pnpm types`, and `pnpm test:unit --maxWorkers=2`. The Vitest result cache records
  **581 test files, no failures**. Existing warning output remains. All new tests
  are enabled; no coverage exclusions or thresholds were changed.
- Current local Cobertura coverage is 100% for TrialCard, TrialStatus, and the
  cache-update helper, 94.59% for useTrialCard, and 92.85% for checkout-return
  confirmation. A comparison of instrumented added lines against merged dev
  `a242bc9392a8a062cce48a7913b57a833b3b909b` found **120/124 covered (96.77%)**;
  that is a local estimate, not a claim that the next Codecov check has passed.
- Billing-auth dependency #14226 remains open at
  `5de13ac07cd3f9759a2ab02d3b0fe79f965f6056`. The new tests cover trial data and
  late trial mutations, not the broader billing/credit surfaces in that dependency.
- Stripe CLI access was refreshed read-only and remains valid for AutoGPT. The
  hosted test session is still sandbox-only and open. Native Safari control is
  now available through Cua without enabling browser debugging; completion and
  backend fulfillment are being verified separately from these frontend tests.

## Hosted Checkout and customer-default card checkpoint

- Completed the owned AutoGPT sandbox Checkout through native Safari using Cua
  and Stripe's published 4242 test card, with no real card data or live charge.
  The session above is now `complete`, `livemode=false`, amount due/paid today 0,
  subscription `sub_1UC8zcEVwK4k8ivIspFi5QqV`. The $50/month, seven-day offer
  remains a test fixture, not the chosen launch offer. This supersedes the
  earlier hosted-interaction limitation; no browser debugging was enabled.
- Before backend confirmation, the local user remained NO_TIER with unconsumed
  eligibility despite Stripe Checkout being complete. Calling the real backend
  confirmation granted TRIAL with a verified card and no pending setup. Repeated
  confirmation retained the same enrollment and subscription. Stripe's initial
  paid $0 `subscription_create` invoice did not incorrectly grant a paid tier.
- Activation and repeated confirmation produced **zero credit transactions**.
  Calling the real onboarding-completion function twice then produced exactly
  one **300-credit** grant and balance 300, keyed by the existing
  `REWARD-{user_id}-ONBOARDING_COMPLETE` identity. No credit-model or notification
  function was mocked in this sandbox/local-database check. Earlier-recipient
  no-top-up behavior also remains covered by the disposable database suite.
- Scheduling cancellation through the test Stripe API, then running actual
  backend reconciliation, retained trial access through the original end date.
  Repeated cancellation kept notification revision 1. This does not prove the
  authenticated cancellation HTTP route or rendered app behavior. The owned test
  subscription is left scheduled to cancel rather than convert to a test charge.
- Moving the already-verified test card to the customer's billing default and
  clearing the subscription-specific default reproduced a real bug: the backend
  removed trial access although Stripe still had the valid effective card.
  Reconciliation now reads the customer's expanded invoice default only when
  neither a subscription payment method nor a legacy subscription source exists.
  Subscription precedence, card type/expiry, pending setup, and ownership checks
  remain enforced. Customer lookup failures do not commit a new entitlement.
- After the fix, that same customer-default card restored the original trial.
  Clearing the effective billing default changed the local user to NO_TIER;
  restoring it restored TRIAL. All transitions preserved consumption, end date,
  cancellation, and the single onboarding grant. The final Stripe and local
  snapshots both report a verified effective card and active trial.
- Added **13 card-fallback regression cases**, including three reproduced
  expected failures before implementation; all expected-failure markers were
  removed. Shared fulfillment fixtures avoid duplicating test setup. Final
  verification: **249 selected trial/billing/onboarding tests**, **29 disposable
  database tests**, whole-backend formatting/type checks, and `git diff --check`
  passed. An earlier selected run was interrupted after 104 passing cases because
  its Redis settings were not pinned; the completed rerun used the owned local
  database and Redis containers explicitly. Existing dependency/test warnings
  remain, with no suppressions or gates changed.
- CI at preceding head `2905c3948379d1d689532b8d42fe5e2350d76d7d` completed with
  no pending or failed checks. Final frontend patch coverage was **83.06% / 70%**
  and backend **85.50% / 80%**. The earlier lower frontend number was an interim
  result before its unit-test upload. This evidence does not cover the newer
  customer-default fix until its own CI completes. The PR remains draft with
  independent review required.
- Still unverified: hosted declines/3DS and actual app return/auth flow; real
  webhook delivery and the deployed API version; portal-driven card replacement;
  legacy-source handling beyond fail-closed behavior; real email delivery/rendering
  and suppressed-notice recovery; concurrent/in-flight spend guarantees; broader
  billing account isolation; final offer/conversion choices, configuration,
  staging, and the explicitly approved live rollout. Nothing is live.

Stripe references: [published test cards](https://docs.stripe.com/testing),
[customer and subscription payment defaults](https://docs.stripe.com/payments/checkout/subscriptions/update-payment-details),
and [subscription payment-source precedence](https://docs.stripe.com/api/subscriptions/object).

## Automatic billing-card webhook checkpoint

- Found that customer billing-default changes and payment-method detachment were
  not routed to trial reconciliation. Added customer updated/deleted, payment
  method attached/updated/automatically-updated/detached, and SetupIntent
  succeeded/requires-action/setup-failed/canceled handling. Typed event models
  identify both current and previous customers; the latter is required when a
  detached card's current customer becomes null.
- These signed events trigger current Stripe reads, not entitlement decisions
  from potentially stale event fields. Lookup is limited to consumed,
  unconverted enrollments whose current user/customer mapping still matches;
  ENTERPRISE and unrelated customers are excluded. Subscription/customer/user/
  enrollment ownership is checked again before reconciliation. Keyset pagination
  processes all targets, and one failed target does not starve later targets.
  Failures return a retryable error instead of acknowledging incomplete work.
- The state-only billing refresh deliberately runs outside the existing event
  claim shortcut: a claim left by a crashed request cannot discard a replayed
  card removal/restoration. Repeated events safely reconcile current state; they
  do not issue trial credits. Existing payment/checkout event handling is unchanged.
- Added a customer-plus-ID lookup index in
  `20260905030000_index_trial_billing_customer`. The local disposable database was
  prebuilt without `_prisma_migrations`, so `migrate deploy` correctly refused to
  baseline it (P3005). No reset or fake history was applied. Executed only the new
  additive SQL with the database explicitly pinned to `127.0.0.1:15432/trial_test`
  and verified the exact index. Full deployment migration validation remains a
  staging gate, separate from this additive local check.
- Reproduced **12 expected failures** before the webhook change, then removed
  every expected-failure marker. Added signed HTTP tests, event-identity and
  failure tests, and real database checks for ownership/exclusions and 101-row
  pagination. Final selected validation: **302 billing/trial/database tests**
  (including all 31 disposable database cases) and **50 webhook/trial API tests**
  passed. Whole-backend formatting/type checks and whole-schema Prisma formatting
  passed; existing test/dependency warnings remain. No checks were weakened.
- Exercised actual Stripe CLI test-mode forwarding to a guarded localhost server
  running the real FastAPI router, real Stripe SDK reads, and local database/Redis.
  The project's FastAPI CLI extras are absent, so the test used its existing
  Uvicorn server without adding production dependencies. The forwarding secret
  stayed in process memory and was not printed or persisted. Both owned test
  processes were stopped after validation; no stored webhook destination changed.
- Clearing and restoring the owned sandbox customer's effective billing default
  automatically revoked/restored trial access via actual forwarded HTTP requests,
  with no manual Checkout confirmation. A captured older removal event was then
  replayed with a fresh local test signature and could not undo the restored card.
  An invalid signature returned 400 without changing the active trial.
- Also created a disposable `tok_visa` test PaymentMethod, made it the test billing
  default, and detached it. Actual event
  `evt_1UC9ixEVwK4k8ivI7tadN3gI` used API `2023-10-16`, with current customer null
  and previous customer `cus_VCUA4PuWBWfp1O`. Its delivery revoked trial access;
  restoring the original verified card restored access. The final run observed
  six HTTP 200 deliveries (including local replay) and one expected HTTP 400.
  No live card or charge was used. The temporary PaymentMethod remains detached.
- Final state is the original TRIAL enrollment/subscription, same consumption
  timestamp and end date, cancellation revision 1, and exactly one 300-credit
  onboarding grant. The original test subscription is still scheduled to cancel.
  This verifies selected test-mode webhook delivery, not production event
  subscription, hosted 3DS/declines, the authenticated app return flow, or email
  delivery after card restoration.
- CI completed at preceding head
  `56a27f6af8b66ab5fdd9ff7ad6ce4346adcb8073` with no pending/failed checks. The new
  webhook checkpoint still needs its own CI and independent review. Analytics
  PRs #14290 (`b3700b0e1e923540faee82f0fcca86587508ba52`) and #14287
  (`d302ac9e9b96f42a7fc6df49c15c004dba35851d`) remain open, as does billing
  isolation #14226 (`5de13ac07cd3f9759a2ab02d3b0fe79f965f6056`, BLOCKED).
- Remaining launch work includes hosted authentication/decline cases, card and
  trial freshness/recovery audits, suppressed email recovery and real email/UI
  validation, concurrent/in-flight spend guarantees, final offer/conversion
  decisions, event/flag configuration, staging migration/runtime checks, and the
  explicitly approved live rollout. Enrollment remains off.

Stripe reference: [billing-card and SetupIntent event types](https://docs.stripe.com/api/events/types).

## Recoverable email suppression and transport checkpoint

- Confirmed that a temporarily suppressed trial notice was terminal even when
  its card/subscription state became applicable again. Recovery now discovers
  eligible suppressed notices and atomically rearms their original outbox row.
  It preserves payload, semantic identity, provider identity guards, and the
  cumulative attempt count. Concurrent requests cannot overwrite accepted work
  or an owned lease. Exhausted suppression recovery becomes a visible failure;
  it never resets the eight-attempt limit.
- Delivery now distinguishes temporary suppression from an obsolete immutable
  payload, ownership, checkout attempt, date, or revision. Obsolete messages stay
  terminal, while temporarily inapplicable messages can be rechecked. Provider
  acceptance reconciliation still precedes another send on retries. The boolean
  freshness interface used by existing notification code remains compatible.
- Added `20260905031000_trial_notice_obsolete_state`, explicitly expanding the
  existing database CHECK constraint. It preserves all prior allowed states and
  data. As with the prior local index check, only this migration SQL was applied
  to the pinned disposable database; no migration-history baseline was invented.
  Deploying the migration and compatible database/notification services before
  enabling enrollment remains a staging/rollout requirement.
- Reproduced four database and two worker/timing expected failures before fixes,
  then removed every expected-failure marker. A new RPC integration test uses
  the actual generated DatabaseManager endpoint models and verifies payload
  round-tripping, reactivation, obsolete completion, and rejection of an unknown
  status. FastAPI guidance informed the typed service-boundary validation.
- An outdated reminder event can no longer send outside the current three-day
  reminder window. Delivery applicability and Python recovery share that window;
  the database candidate selection uses the same three-day policy.
- Real transport validation used an isolated RabbitMQ 4.1.4 container on
  `127.0.0.1:15772`, an actual DatabaseManager HTTP service on
  `127.0.0.1:18044`, the production notification consumer acknowledgment method,
  and the real trial delivery callback. The consumer had no direct database
  connection and used actual RPC. Stripe reads/updates used only the owned test
  subscription. The sole substituted boundary was an in-process recording
  sender: no Postmark request or external email was made by this transport test.
- A queued welcome was suppressed while the test trial was canceled. Removing
  cancellation made it applicable, and the actual producer reactivated the same
  notice. It was recorded once at cumulative attempt 2; two duplicate broker
  wakeups did not record another send. Payload and ID stayed unchanged. A separate
  outdated-payload notice crossed RPC into `obsolete` without being sent. The
  original welcome ID is `f0028744-e5d7-44fc-8530-ccfff43a6133`; its local recording
  identifier is explicitly not a real provider acceptance.
- Cleanup restored the original test cancellation and email-verification setting.
  The user remains TRIAL with balance 300 and the original consumption/end dates;
  notification revision is now 3 because the test canceled/resumed/canceled.
  The database service stopped, and all broker queues/unacknowledged counts were
  zero before stopping the task-owned broker. Docker reports it stopped with exit
  137 and `OOMKilled=false`; logs show SIGTERM and stopped message stores. This is
  not a claim that broker graceful-shutdown/restart behavior is fully validated.
  The stopped broker and local test records are retained for follow-up evidence.
- Final local checks: **315 selected billing/trial/database/RPC tests** (including
  all **44 disposable database cases**) and all **230 notification tests** passed.
  Whole-backend formatting, generated Prisma stubs, pyright, and `git diff --check`
  passed. Existing dependency/test warnings remain; no gates or suppressions were
  relaxed. CI at preceding head `6f342300ea91598d08b7dbf4228135587da99b2f` finished
  with no pending or failed checks; this checkpoint still needs its own CI/review.
- Rendered all seven trial messages locally to
  `/private/tmp/autogpt-trial-email.xROhx7/rendered/index.html`. Their $50/month,
  seven-day values remain fixtures. These are generated HTML/text previews, not
  a completed desktop/mobile/email-client visual review.
- Postmark identity was checked read-only: server 15280758, AutoGPT Platform,
  DeliveryType Live. A proposed validation using `POSTMARK_API_TEST` was rejected
  by auto-review because it would transfer rendered contents and metadata to an
  external destination without explicit approval. That command did not execute,
  and it was not retried or bypassed. Real metadata lookup, provider acceptance,
  inbox delivery, and provider-reconciliation recovery remain unverified. User
  approval for those transfers and a test inbox address have been requested.
- Remaining work includes visual/email-client QA, approved provider/inbox tests,
  failed-delivery operational recovery and retention, broker/service restart and
  deployment behavior, hosted 3DS/declines, trial freshness and concurrent/in-flight
  spend guarantees, final offer/conversion choices, configuration, staging, and the
  explicitly approved live rollout. Enrollment remains off; nothing is live.

Postmark reference: [documented email testing modes](https://postmarkapp.com/support/article/1213-best-practices-for-testing-your-emails-through-postmark).

## Phase-one spend policy decision — 2026-09-06

- The user explicitly accepted soft caps for phase one after discussing parallel
  admission, delayed cost reporting, and a final provider request crossing the
  allowance. Daily, weekly, and lifetime allowances remain configurable. Recorded
  exhaustion blocks subsequent work; in-flight work can overshoot. There is no
  verified dollar or percentage upper bound on that overshoot.
- Strict reservations and provider-admission enforcement are not launch gates
  for this phase. Earlier checklist references to concurrent/in-flight spend
  guarantees are superseded by this decision, not evidence that such guarantees
  have been implemented. Accurate accounting and existing access checks remain
  in scope.
- These limits concern AutoGPT's underlying Copilot costs, not customer overage
  charges. This decision adds no card charges, no extra onboarding grant, and
  does not approve final offer values, subscription-conversion policy, production
  configuration, external email testing, or a live rollout.
- Anthropic documents that its SDK budget stops a run after spend exceeds the
  limit; it is not an exact pre-request spending reservation. SDK cost fields
  are estimates rather than authoritative invoices. We will not present either
  the SDK guardrail or our pre-turn checks as a hard financial ceiling.

References: [SDK budget behavior](https://platform.claude.com/cookbook/claude-agent-sdk-scheduled-repository-reviewer-scheduled-repository-reviewer),
[SDK cost-accounting scope](https://code.claude.com/docs/en/agent-sdk/cost-tracking).

## Delayed foreground cost attribution checkpoint

- Reproduced a lost-cost bug: a foreground trial turn could finish, then report
  its cost after the user became paid or lost trial access. Accounting consulted
  the current tier and current trial status, dropping that cost from the trial's
  lifetime total. Two executor regressions and seven database cases were written
  before the fix; the executor reproduction showed zero trial writes instead of
  the required one. All expected-failure markers have been removed.
- The common foreground executor now captures the active, consumed trial once
  before starting the engine. Its immutable, user-scoped context propagates into
  the heartbeat driver and asynchronous descendants, including delayed cost
  reconciliation. Cleanup restores the previous context even on cancellation
  or errors. A subsequent paid turn in the same chat gets a separate non-trial
  snapshot and cannot add its spend to the earlier trial.
- Explicit trial-cost writes atomically match both user ID and consumed trial
  ID, independently of the trial's later status/card state. Wrong-user,
  nonexistent, and unconsumed attribution is rejected. Existing unscoped callers
  retain their previous behavior. Cost-log metadata now includes the captured
  `subscription_trial_id` (null for an explicitly non-trial foreground turn).
- Validation passed: **238 Copilot/context/executor/rate-limit/token-tracking
  tests**, plus **18 disposable PostgreSQL enrollment/cost/RPC tests**. The final
  focused 12-test rerun uses the real heartbeat/background-task boundary and
  verifies matching cost-log attribution. FastAPI guidance informed tests of the
  actual generated database endpoint models, including 400 ownership rejection
  and 422 malformed-ID validation. Whole-backend formatting, generated Prisma
  stubs, pyright, and diff checks passed. Existing dependency/test warnings remain.
- The first broad test attempt hit sandbox-denied localhost access and was
  interrupted; it is not counted as a pass. The complete rerun with approved
  access to the existing local Redis and disposable PostgreSQL services passed.
- No schema migration is needed for this checkpoint. Deploy the compatible
  database service before updated executors: older generated endpoint models
  do not consume the new optional attribution field. CI at preceding head
  `827011634d1845b0a911da5a3f76423cff9ce5be` completed with no outstanding/failed
  checks; the new commit still requires its own CI and independent review.
- Scope limits: this is not a reservation, retry-deduplicated settlement ledger,
  or crash-safe cost queue. Standalone background jobs outside the foreground
  executor still use their legacy attribution path. Missing provider costs,
  process loss before persistence, best-effort cost logging, and daily/weekly
  settlement-time windows remain accounting/measurement limitations to assess;
  phase-one soft-cap acceptance does not make these hard guarantees.
- No real provider request, email, Stripe mutation, production configuration, or
  deployment occurred in this checkpoint. Trial enrollment remains off.
