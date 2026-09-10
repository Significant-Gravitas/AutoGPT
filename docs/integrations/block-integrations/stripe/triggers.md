# Stripe Triggers
<!-- MANUAL: file_description -->
Blocks that trigger on subscription lifecycle events delivered by Stripe webhooks.
<!-- END MANUAL -->

## Stripe Subscription Trigger

### What it is
Triggers on Stripe subscription events (new, upgrade, cancel). Uses Stripe webhooks directly — real external customers only, no internal or demo account noise.

### How it works
<!-- MANUAL: how_it_works -->
Connect a Stripe API secret key (`sk_live_...` or `sk_test_...`) with permission to
manage webhook endpoints. The platform registers a webhook endpoint in your Stripe
account for the events you selected and stores the signing secret Stripe returns.
Every incoming delivery is checked against that secret — timestamp plus HMAC-SHA256
over the raw body, with a five-minute replay window — before the block fires.

Plan details are read from the subscription's first item price, falling back to the
top-level `plan` object on older Stripe API versions. The endpoint is deleted from
your Stripe account when the last trigger using it goes away.

Deliveries that fail verification never reach the block: a missing or malformed
`Stripe-Signature` header, a signature that doesn't match, or a timestamp outside
the five-minute window are all rejected with `403`, and an event with no `type` is
rejected with `400`. If a delivery passes verification but its subscription object
can't be parsed, the block fails with a parse error rather than emitting partial
outputs.

Endpoints are shared, not one per trigger: the platform reuses an existing webhook
whose registered events already cover the ones you asked for, keyed on your
credentials. Stripe caps an account at 16 endpoints, so that ceiling is reached
only with many distinct event-filter combinations on the same key, not with many
triggers.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| events | Subscription lifecycle events to subscribe to. Cancellation and churn workflows need `deleted`, which is off by default. Note that `updated` is high-volume — Stripe sends it for any change to the subscription, including renewals, payment-method changes and metadata edits, not just upgrades. Use `previous_attributes` to tell an upgrade from routine churn. | Events | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the payload could not be parsed | str |
| payload | Full Stripe event object as received from the webhook | Dict[str, Any] |
| event_type | Stripe event type, e.g. customer.subscription.created | str |
| subscription_id | Stripe subscription ID (sub_...) | str |
| customer_id | Stripe customer ID (cus_...) | str |
| status | Subscription status: active, trialing, past_due, canceled, etc. | str |
| cancel_at_period_end | True if the subscription is scheduled to end when the current billing period does, rather than having ended already | bool |
| canceled_at | Unix timestamp of when the subscription was canceled, or 0 if it has not been canceled | int |
| previous_attributes | On `updated` events, the changed fields' prior values, as sent by Stripe. Empty for other events. Compare against the subscription in `payload` to tell an upgrade from a renewal — e.g. a key of `items` or `plan` means the plan itself changed. | Dict[str, Any] |
| plan_name | Nickname of the subscription's first item price. Prices without a nickname fall back to the raw price ID (price_...). Only the first item is read; see `payload` for multi-item subscriptions. | str |
| plan_interval | Billing interval of the first subscription item: day, week, month or year | str |
| amount_cents | Unit amount of the first subscription item, in the smallest currency unit — cents for USD, but whole units for zero-decimal currencies like JPY and KRW. This is not the subscription total when there is more than one item. | int |
| currency | Three-letter ISO currency code | str |
| livemode | True for live Stripe data, False for test mode | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Revenue Notifications**: Post to a team Slack or Discord channel whenever someone subscribes or upgrades, wiring `customer_id`, `plan_name`, and `amount_cents` into the message. Because the events come from Stripe rather than an internal database, only real paying customers are counted.

**Onboarding Sequences**: Kick off a welcome email series on `customer.subscription.created`, branching on `plan_name` so each tier gets the setup steps that apply to it.

**Churn Recovery**: Start a win-back workflow on `customer.subscription.deleted`, using `cancel_at_period_end` to tell a scheduled end (still time to intervene) from an account that has already lapsed.
<!-- END MANUAL -->

---
