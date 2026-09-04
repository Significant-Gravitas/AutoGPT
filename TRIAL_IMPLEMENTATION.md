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
- Revalidate the complete current migration (including TRIAL_UPDATE), full backend
  CI, complete frontend checks, and exact-head PR review. No PR has been opened.
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

## Outstanding launch choices

The final offer amounts/duration/price and launch cutoff remain unset pending
unit-economics data. The initial trial config must remain off until these values
are chosen and the complete integration is verified. Onboarding credits are a
separate wallet from the operator-side chat spend budget.
