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
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| events | Subscription lifecycle events to subscribe to | Events | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the payload could not be parsed | str |
| payload | Full Stripe event object as received from the webhook | Dict[str, Any] |
| event_type | Stripe event type, e.g. customer.subscription.created | str |
| subscription_id | Stripe subscription ID (sub_...) | str |
| customer_id | Stripe customer ID (cus_...) | str |
| status | Subscription status: active, trialing, past_due, canceled, etc. | str |
| plan_name | Plan nickname from the subscription's first item price | str |
| plan_interval | Billing interval: month or year | str |
| amount_cents | Plan unit amount in the smallest currency unit (e.g. cents for USD) | int |
| currency | Three-letter ISO currency code | str |
| livemode | True for live Stripe data, False for test mode | bool |

### Possible use case
<!-- MANUAL: use_case -->
Post a message to a team Slack or Discord channel whenever someone subscribes or
upgrades, wiring `customer_id`, `plan_name`, and `amount_cents` into the notification.
Because the events come from Stripe rather than an internal database, only real
paying customers are counted.

Other uses: kick off an onboarding email sequence on `customer.subscription.created`,
or start a win-back workflow on `customer.subscription.deleted`.
<!-- END MANUAL -->

---
