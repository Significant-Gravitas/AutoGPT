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
  strict reservations. Audit concurrent/in-flight turns, provider caps, late
  settlement, background work, BYOK, and failure behavior before promising any
  hard spend ceiling.
- Notification reliability: Redis claim/queue crash window, missed-reminder
  recovery, removed-card notices, queue consumer retries, and visual email QA.
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
