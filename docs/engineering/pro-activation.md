# Trial conversion to Pro or Max and usage

This backend change is stacked on allowance PR #15184, tested parent
`223b3f1ac9937fb8ade0d87e792ae36a3357b204`. It gives an eligible trial's first settled
conversion to Pro or Max fresh daily and weekly consumption. Trial exhaustion
never initiates a purchase. The trial's original accepted schedule can convert automatically;
ending it early requires the separate explicit confirmation below.

## Durable activation and accounting

An owned, live Stripe subscription must be active, with an authoritative settled
recurring invoice for that subscription. A checkout redirect, completed Checkout
Session, trial setup invoice, subscription tier, or pending PaymentIntent does
not establish readiness. Settlement can include a valid discount or customer
credit; cash `amount_paid > 0` is not required.

`PaidUsageActivation` records one initial trial-conversion subscription/invoice identity per user.
The user row is locked before the trial row. Entitlement, conversion identity,
and the completed generation are committed together after Redis can read the
new generation's calendar counters. No daily or weekly key is deleted or zeroed.
A repeated reconciliation probes the same generation, preserving any paid usage
already written to it. Failed transactions can retry without destructive cleanup.

The generation changes the namespace of the existing UTC daily and ISO-week
counters. Configured Pro and Max limits and reset times stay the same. Work captures its
billing origin before execution and propagates it into background and child work.
Delayed trial costs update the original lifetime trial ledger and retain their
original usage attribution. Neither conversion nor recovery clears trial history,
credits, runs, conversations, or historical costs.

Only a consumed, owned trial that has not already converted can qualify for a new
generation. A tier label alone cannot establish that history. These transitions
have explicit behavior:

| Transition | Fresh daily and weekly usage |
| --- | --- |
| Unconverted trial → first paid Pro | Yes, after settled payment and completed reconciliation. |
| Unconverted trial → first paid Max | Yes, after settled payment and completed reconciliation. |
| Paid Pro → Max | No; preserve consumption. |
| Paid Max → Pro | No; preserve consumption. |
| Unpaid account without a consumed trial → paid signup | No; preserve consumption. |
| Previously converted trial, renewal, returning paid subscriber, or admin tier edit | No new activation. |

Retries of a qualifying conversion finish the same activation and retain all paid
usage already written. The migration records the policy start in
`PaidUsageActivationPolicy` at Stripe's whole-second precision; pre-policy
payments are not retroactively reset.
An established paid-plan change can reconcile from a settled proration invoice
with owned recurring service at the current price and qualifying prior paid
history. Prorations never establish the initial-payment evidence for a reset.
Authoritative database reads bypass process-local tier caches for enforcement.
If entitlement or accounting state cannot be established, execution fails closed.

## Authenticated API

All routes below require the existing user bearer authentication and subscription
status rate limiter. Attempt IDs are always scoped to the authenticated user.
Foreign IDs return 404. The exported OpenAPI schema and generated client include
these endpoints. The historical `pro-activation` route and storage identifiers
remain compatible while supporting both Pro and Max. This PR contains no visual redesign.

| Method and exact endpoint | Request | Effect |
| --- | --- | --- |
| `POST /api/credits/pro-activation/preview` | `{"plan":"MAX","return_to":"/chat/THREAD?resume=1"}` | Preview conversion of the user's existing unconverted trial to Pro or Max and persist quoted terms. Never starts payment. |
| `POST /api/credits/pro-activation/{id}/confirm` | `{"confirmed":true,"terms_token":"HEX_SHA256"}` | Record explicit consent, then end the same trial subscription early with a stable idempotency key. |
| `GET /api/credits/pro-activation/{id}` | None | Retrieve live payment state and retry entitlement/usage reconciliation. Never creates or retries a charge. |
| `GET /api/credits/pro-activation/current` | None | Recover the user's persisted attempt after navigation, or reconcile their existing paid Pro/Max subscription. Never creates or retries a charge. |

Preview accepts an application-relative `return_to` path, including its query and
fragment; external destinations are rejected. The default is `/settings/billing`.
The optional `plan` is `PRO` or `MAX`; omitting it retains the trial's accepted
plan when that offer is Pro or Max. Other accepted trial offers require an
explicit Pro/Max target for this flow. Keeping the original plan preserves its
originally accepted price even if the current offer has changed. Selecting another
plan previews its configured price in the trial's accepted currency and billing
interval and requires fresh explicit confirmation of those terms. A saved preview
retains its selected price through confirmation and recovery even if the catalog
changes; expiry and charge, discount, and tax revalidation still apply. A fresh
preview uses the current catalog. The trial's original offer remains in its
lifetime history. A confirmed request changes the existing subscription item,
never creates a second subscription, and uses exactly the item/price previewed.
Once confirmed, the persisted destination and accepted terms are immutable.
Initial paid signup continues through the existing authenticated
`POST /api/credits/subscription` and Stripe Checkout, which presents the charge and
requires payment confirmation. Signup without a consumed trial does not reset
consumption; successful recovery reports `usage_reset=false`. Its validated success
destination is also saved in subscription metadata for `/current` recovery. Destinations exceeding Stripe's
metadata length limit are recovered from the owned Checkout's success URL after
checking the application origin.

Every successful activation response has these fields. Optional values are JSON
`null`; `return_to` always has a value and `usage_reset` defaults to false:

| Field | Type / meaning |
| --- | --- |
| `id` | Attempt ID, or null for initial paid signup without an early-conversion attempt. |
| `status` | One of the states in the next table. |
| `terms` | The persisted commercial terms below, or null for the existing Checkout flow. |
| `terms_token` | SHA-256 of the exact persisted terms, required by confirm; otherwise null. |
| `return_to` | Validated application-relative return destination. |
| `invoice_id` | Authoritative current invoice ID, when available. A `ready` response uses the invoice verified during reconciliation, even if it changed after the initial status read. |
| `hosted_invoice_url` | Stripe's hosted URL for the existing invoice when payment/action is required. |
| `retry_after_seconds` | Suggested status polling delay, normally 3 while processing. |
| `error_code` | Machine-readable diagnostic, otherwise null. |
| `usage_reset` | True only for `ready` when this subscription's current invoice and price match its completed initial trial conversion. False for nontrial signup, renewals, paid-plan changes, and returning/pre-policy/legacy conversions without an activation. It describes the original completed conversion; this poll never resets consumption. |
| `activation_id` | Stable completed initial activation ID when `usage_reset` is true; otherwise null. Repeated polling returns the same ID and never resets counters again. |

Before showing confirmation, display the commercial `terms`: `plan` (`PRO` or `MAX`),
`amount_due` formatted in `currency`, `billing_interval`
(`month` or `year`), `billing_interval_count`, `charge_timing`
(`on_confirmation`), `renewal_unit_amount`, and `renewal_terms`.
`price_id` identifies the selected paid price. `accepted_offer_token` binds the
conversion to the originally accepted trial offer without rewriting that offer;
`expires_at` is the UTC quote expiry, 15 minutes after preview.

Also display `renewal_discounts`: each contains `amount_off`, `percent_off`,
`currency`, `duration`, `duration_in_months`, and `ends_at` (Unix seconds when
available). A `once` discount applies to the initial invoice, not every renewal.
Display `renewal_tax.automatic`, `renewal_tax.price_tax_behavior`, and any explicit
`renewal_tax.rates` (`display_name`, `percentage`, `inclusive`, `country`, `state`).
These modifiers are included in the token and revalidated even when today's
amount happens to be unchanged. `amount_due` comes from a Stripe invoice preview
with the existing subscription's discounts, taxes, and customer credit. The base
renewal amount and modifiers explain subsequent charges without promising a
fixed future tax amount or an indefinitely recurring introductory discount.

| State | Client behavior |
| --- | --- |
| `confirmation_required` | Show the complete terms. Send confirm only after an explicit user action. Missing, false, string, or numeric confirmation is rejected. |
| `payment_required` | Open `hosted_invoice_url` to pay or update the payment method for this invoice. Retain the attempt and destination; poll status after return. Do not open another Checkout. |
| `action_required` | Complete authentication using the same hosted invoice. Then poll status; authentication itself is not readiness. |
| `processing` | Keep the activation recoverable and poll GET using the delay. Do not announce failure or fresh allowance. A settled payment can remain here while database/Redis reconciliation recovers. |
| `ready` | Paid Pro/Max entitlement and the applicable reset decision are authoritative. Refetch the three data sets below, then continue to `return_to`. Announce an initial fresh allowance only when `usage_reset=true`, once per `activation_id`. |
| `not_applicable` | The owned live subscription is outside Pro/Max (`not_paid_activation_subscription`), or its paid price has changed away from this attempt's accepted target (`plan_changed`). Stop activation polling, refetch subscription/usage, and continue to the preserved `return_to`. This is not a failed payment and grants no new reset. |
| `failed` | An authoritative canceled/expired subscription or void/uncollectible invoice has ended this attempt (`payment_canceled`). No fresh usage is granted. |

401/403 require authentication; 404 means no owned attempt/subscription was found.
409 means the unconfirmed terms expired/changed, ownership cannot be established,
or another confirmation currently owns the account checkout lock. Fetch current
state before deciding whether a fresh preview is appropriate. 422 means invalid
input. A preview/terms-validation Stripe failure can return 502 before payment is
submitted. After durable consent, an uncertain mutation or reconciliation result
returns `processing`, not a payment-failure claim. The current-status route also
returns `processing` during database/Stripe outages.
An unknown price or ambiguous subscription items remain `processing` with
`error_code=plan_unavailable`; a stale local tier alone never makes paid Pro/Max
recovery fail. Accepted trial prices remain recognized after offer changes.
If the plan changes during reconciliation, readiness waits for the locked tier,
current subscription price, and attempt's accepted target to agree. A subsequent
poll observes any changed attempt target as `not_applicable` (`plan_changed`).
`/current` skips subscriptions outside Pro/Max while looking for an applicable paid
subscription, returning `not_applicable` if no applicable subscription exists.

On refresh/navigation, call `/current`; do not derive success from URL parameters,
an old tier response, or a cached completed Checkout. If the original confirmation
request had no definitive response, it can be retried with the **same** persisted
attempt ID and terms token. A live subscription that already left `trialing` is
only reconciled. If still trialing, the exact Stripe request uses the same key
`pro-activation:{id}`; this is recovery of existing consent, never a new purchase.
Routine polling uses GET exclusively. After 23 hours the backend stops replaying
an ambiguous mutation (`processing`, `error_code=recovery_required`), before
Stripe's minimum idempotency retention expires; an operator must establish its
outcome. Never create a replacement attempt or invoice to resolve that state.
If accepted terms change after consent was persisted, the backend also stops
submission (`processing`, `error_code=terms_changed`); establish the existing
payment outcome before arranging any new consent.
An unconfirmed trial whose live accepted price changed also returns `terms_changed`
instead of offering confirmation against stale terms. Refresh the preview only
after the subscription again matches its accepted offer; no payment was submitted.

In the final five minutes before the live scheduled trial end, confirmation of
the original plan returns processing and lets the already accepted scheduled
conversion happen. Preview or confirmation of a different plan is rejected with
409 in this window; it must not silently charge the original plan after showing
the other plan's terms. A previously confirmed alternate-plan attempt that reaches
this window without a definitive mutation outcome returns `processing` with
`error_code=trial_ending`; poll its existing state rather than retrying a charge.
This avoids racing a scheduled Stripe transition with an early-end request.

After `ready`, invalidate and **await refetch** of queries for:

- `GET /api/credits/trial` (active/converted state and preserved lifetime history).
- `GET /api/credits/subscription` (entitlement and billing state).
- `GET /api/chat/usage` (the applicable Pro or Max allowance and current consumption).

Use the generated query-key helpers for these exact endpoints; discard any old
paywall/exhaustion decision derived from their previous results. Preserve
`return_to` throughout hosted payment/authentication navigation.
`usage_reset=true` describes the completed initial invoice, not an operation
performed by this poll. Preserve the last announced `activation_id` across
navigation. Later renewal or plan-change invoices return false and no activation
ID; repeated recovery of the initial invoice must not repeat the fresh-usage claim.

## Rollout and recovery requirements

Pause new work and checkout/early-conversion admission, then drain all foreground,
queued, background, batch, and child-task work from old workers **before applying
the migration**. The migration records the policy start time: applying it while
old workers can still admit paid work would allow a later reconciliation to
exclude that work from the fresh allowance. Install the schema, roll out API,
reconciliation services and every worker/producer together, then resume admission.
Scheduled Stripe payments during this window reconcile after the new code starts.
Legacy batch callbacks without a stored accounting context fail closed, mark the
job errored, and release the original lock without billing under a new context.
They must still be drained before rollout. Mixed old/new usage writers must not account newly
admitted work after activation. Keep the
generation-aware accounting code during rollback for accounts already activated;
rolling back to readers of only the old Redis keys would lose correct enforcement.

Subscribe Stripe webhooks to `invoice.paid`, payment-success/failure events, and
subscription changes. Leave the periodic Stripe reconciliation sweep enabled.
Webhook retries, status polling, and the sweep all converge on the same database
activation identity. Recovery must never DELETE usage keys, clear an activation
row, clear the lifetime trial ledger, or call a billing mutation merely because
the allowance is exhausted. Retry a paid-but-unreconciled activation through GET
or the reconciliation function after the dependency recovers.

Before enabling the future frontend action, validate the pinned Stripe API using
test-mode trials/test clocks: exhausted and partial early conversion to Pro and Max,
scheduled conversion, Pro↔Max changes preserving consumption, nontrial signup
without a reset, SCA, declines, 100% discounts/customer credit, product-restricted
discounts when changing plans, duplicate events, and a
lost mutation response. Local tests use Stripe boundary doubles and real
disposable PostgreSQL/Redis for transaction/concurrency/accounting behavior; they
do not exercise production billing. No rollout or production data change is part
of this PR.

Stripe references: [subscription update and payment behavior](https://docs.stripe.com/api/subscriptions/update),
[invoice previews](https://docs.stripe.com/api/invoices/create_preview), and
[invoice payment relationships](https://docs.stripe.com/api/invoice-payment).
Product restrictions are described by the
[coupon object](https://docs.stripe.com/api/coupons/object?api-version=2025-02-24.acacia).
