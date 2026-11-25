# Scripted model

A fake model provider for tests. A Playwright journey drives the real copilot
pipeline (engine, tool routing and execution, persistence, streaming and
resume, cost accounting) while this server plays the model, turn by turn, from
a script. `CHAT_TEST_MODE`'s canned reply skips all of that.

It is also the journeys stack's Stripe: see [Fake Stripe](#fake-stripe).

It only runs where `docker-compose.journeys.yml` is added: CI's journeys leg
and local test stacks. Self-hosted and hosted deployments never start it.

## How the platform reaches it

`docker-compose.journeys.yml` points the chat, memory LLM and memory embedder
base URLs at `http://scripted_model:4010/api/v1`, so both engines and their
side calls come here:

| Caller | Request | Dialect |
| --- | --- | --- |
| SDK engine (the Claude Agent SDK's CLI, the default) | `POST /api/v1/messages` | Anthropic Messages, streamed |
| Baseline engine (`FORCE_FLAG_COPILOT_SDK=false`) | `POST /api/v1/chat/completions` | OpenAI chat completions, streamed |
| Side calls (session title) | `POST /api/v1/chat/completions` | non-streamed, no tools |
| Memory extraction | `POST /api/v1/responses` | OpenAI Responses, structured output |
| Memory search | `POST /api/v1/embeddings` | OpenAI Embeddings |

The CLI also sends `HEAD /api/api/hello` to check it can reach the server,
and may ask `POST /api/v1/messages/count_tokens`; both are answered.

## Scripts

A journey picks a script with a marker in its chat message:
`[[script:<name>]]`, or `[[script:<name>#<nonce>]]` to find its own requests
in the journal. Scripts are JSON files in
`frontend/src/playwright/journeys/model-scripts/`, mounted at `/scripts` and
read on every request (edits apply without a restart). A journey imports its
script to know what to expect.

```json
{
  "description": "What the journey checks.",
  "title": "Answer to side calls, e.g. the chat's title",
  "turns": [
    {
      "text": "Let me check your library first.",
      "tool_calls": [{ "name": "find_library_agent", "input": { "query": "doubler" } }],
      "usage": { "input_tokens": 3000, "output_tokens": 40,
                 "cache_read_input_tokens": 20000, "cache_creation_input_tokens": 0 },
      "stream": { "chunk_words": 3, "delay_ms": 15 }
    },
    {
      "expect_tool_results": [{ "tool": "find_library_agent", "contains": "No agents" }],
      "text": "You don't have one yet. Want me to build it?",
      "usage": { "input_tokens": 3400, "output_tokens": 20 }
    },
    {
      "expect_user_text": "Yes, build it",
      "text": "On it.",
      "usage": { "input_tokens": 3500, "output_tokens": 5 }
    }
  ]
}
```

- **A turn the platform sends itself**, such as an expert's kickoff, has no
  message the journey can type a marker into. When no user message holds
  one, the marker is looked for in the system prompt, so a journey puts it in
  text the user sets that the system prompt carries: an expert's writing
  sample (`<voice_preferences>`). A marker in a user message still wins.
  The sample stays in that expert's system prompt, so every later request
  of that hire with no marker of its own goes to the same script, a turn the
  platform starts later included. Give such a turn its own marker in a user
  message, or expect it to run past the script's end.
- **Which turn answers a request:** turn `n` answers the request whose
  conversation already holds `n` assistant messages after the user message
  that carries the marker. Requests carry the whole conversation, so the
  server keeps no state: concurrent runs of one script, a retried turn and a
  resumed session all land on the right turn. A follow-up the user queued
  during a turn reaches the model after that turn's answer, so it gets the
  next turn.
- **A message with a marker of its own** starts that script: when several
  user messages carry one, the newest picks the script. After a stopped
  turn, the platform may resume the conversation or hand it to the model as
  text inside the next message, so a journey that sends a message after a
  Stop gives it its own marker; either way it gets that script's first turn
  (`copilot-stop-journey.spec.ts`).
- **The newest marker is read from the user's own words.** The platform
  adds tagged blocks to a user message, such as recalled memories
  (`<memory_context>`) or the earlier chat passed on as text
  (`<conversation_history>`). These can quote earlier messages of the chat,
  or of the user's other chats, markers and all, so a marker inside a tagged
  block never outranks one the user typed. A request with no marker in the
  user's own words, such as a title request that quotes the chat inside a
  `<conversation>` tag, takes the first marker in its user messages.
- **Tools:** name them as the baseline engine does (`find_library_agent`).
  The SDK engine offers them as `mcp__copilot__find_library_agent`; the server
  calls whichever the request offers. Most platform tools (`create_agent`,
  `edit_agent` and others) are deferred: on both engines the model reaches
  them through `run_capability` with `{"id": "tool:<name>", "input": {...}}`,
  and `create_agent` needs the building guide read first
  (`{"id": "skill:agent_building_guide"}`). See
  `model-scripts/build-doubler-agent.json`.
- **An argument read from a tool's result:** a model passes on an id it read
  from what a tool returned (`list_schedules`, then `pause_schedule`). In a
  tool call's input, write
  `{"$from_tool_result": {"tool": "run_capability", "pattern": "\"schedule_id\": \"([^\"]+)\""}}`
  in place of the value: it becomes the pattern's first group in that
  tool's results the request sends back, which are only the last turn's:
  of several calls to the tool in that turn, the last one with a match
  whose first group took part (an optional group can match empty-handed).
  An earlier turn's result is never read. No match fails the turn, naming
  the results it saw.
- **Checks** are logged, and a journey asserts on them through the journal:
  `expect_tool_results` against the tool results the platform sends back,
  `expect_user_text` against what the user said since the last answer.
- **Streaming:** text goes out `chunk_words` words at a time, `delay_ms`
  apart.
- **Holding a reply open:** with `"hold_after_words": n` in `stream`, the
  reply stops after `n` words until the journey releases its run with
  `POST /__scripted__/release?nonce=<nonce>` (`releaseHeldReply` in
  `scripted-model.ts`), for example once it has cut the connection. The script
  needs a marker with a nonce. Each release lets one held reply through: one
  sent early lets the reply straight through, and a second held reply in the
  same run waits for a second release. A reply nobody releases finishes after
  two minutes and fails its journal check.

A request the script can't answer fails the turn with a 400 that says why:
no marker, an unknown script, a turn past the script's end (an extra tool
round or retry), or a tool the platform didn't offer. A journey fails loudly
instead of waiting on a reply that never comes. A request that offers no tools
is a side call and gets the script's `title`.

Memory extraction requests get a minimal value for the object, array and scalar
schema forms Graphiti requests. Lists are empty and scalar fields use neutral
values, so the memory ingestion pipeline runs successfully without making up
facts about the test conversation. These calls are journaled as
`openai-responses` side calls. Memory searches get deterministic, nonzero
1536-dimensional vectors like `text-embedding-3-small`; Graphiti consumes its
configured 1024 dimensions. They are journaled as `openai-embeddings`.

## Cost

The server keeps the provider's invoice, priced in `invoice.py` from
Anthropic's list prices (the same as `backend/copilot/anthropic_rates.json`):

- chat completions report it as `usage.cost`, as OpenRouter does, which is
  what the baseline engine charges;
- the SDK engine charges the CLI's own estimate, priced from the same list.
  Message ids are `msg_…`, not OpenRouter's `gen-…`, so the platform doesn't
  try to look the cost up on openrouter.ai.

Each scripted chat-reply entry records `usage` and `cost_usd`, so a journey can
check that the platform recorded exactly what the provider billed.

## Journal

`GET http://localhost:4010/__scripted__/journal?nonce=<nonce>` (or
`?script=<name>`) lists a scripted run's chat requests: dialect, model, which
turn answered it, tool results, expectation checks, usage, cost, and whether
the whole reply reached the platform (`completed`). Refused scripted turns are
there with their reason. Unfiltered reads also include memory side calls and
unexpected requests; memory calls deliberately have no script or nonce, usage,
or billed cost. The container log has one line per request.

Each entry also names the API key the request presented (`x-api-key`, or a
bearer token) as `credential`: the first 16 hex digits of its SHA-256, never
the key itself. `?credential=<fingerprint>` lists the requests made with one
key. The journeys stack points Anthropic clients at this server
(`ANTHROPIC_BASE_URL`), so an AI block run with a key the journey's user saved
lands here too, and a journey can check the block ran once, with that key.

## Fake Stripe

The hosted configuration (`.github/scripts/hosted_journeys_env.py`) points
the backend's Stripe SDK here (`STRIPE_API_BASE=http://scripted_model:4010/stripe`,
a `sk_test_` key and a webhook secret), so a journey buys credits and plans
through checkout and webhooks without a Stripe account. Objects live in
memory for the stack's lifetime, so ids stay the same across create, retrieve
and list: the backend reads Stripe back often. Objects take the shapes of the
API version the platform's SDK pins (`2025-02-24.acacia`): a subscription's
period is on the subscription, an invoice's subscription on the invoice.

**Plans.** The fake starts with the products and prices in
`scripted_model/fake_stripe/catalog.json` for sale, under fixed ids (Pro and
Max, monthly and yearly). A price's `lookup_key` is the key the platform's
`copilot-tier-stripe-prices` flag gives it (`PRO`, `PRO_YEARLY`), and the
journeys stack builds that flag from the same file, so the platform asks for
prices the fake has. Its amounts differ from the frontend's marketing prices
on purpose: a price on a page is the one Stripe charges.

| Platform call | What the fake does |
| --- | --- |
| `Customer.create` (its idempotency key too), `retrieve`, `modify`, `list` | keeps customers |
| `checkout.Session.create`, `retrieve` (with `expand`), `list`, `expire` | keeps sessions; a session's `url` is the fake's checkout page |
| `PaymentIntent.retrieve`, `PaymentMethod.list` | the payment a checkout took, and the card it saved |
| `Price.retrieve` | the catalogue's prices |
| `Subscription.list`/`retrieve` | the subscriptions checkouts started; a list leaves out cancelled ones unless `status` asks |
| `Subscription.modify` with `cancel_at_period_end`, `metadata`, `cancellation_details` | changes it, then sends `customer.subscription.updated` with `previous_attributes` |
| `Subscription.cancel` (`cancellation_details`, no final invoice) | ends it now, then sends `customer.subscription.deleted` |
| `Invoice.list`/`retrieve` | the paid invoices of subscriptions, each with a hosted page and the next number in its customer's sequence (`<prefix>-0001`, `-0002`) |
| `billing_portal.Session.create` | the fake's billing portal for the customer |

Anything else gets Stripe's 404 for an unknown URL and is listed at
`GET /__stripe__/requests?unserved=true&after=<n>`. So is a call to a URL the
fake serves with parameters it doesn't model, refused with a 400 that says
so: a subscription's price change (`items`) or trial end, a final invoice on
cancelling, an inline price or a one-time price on a subscription checkout,
a subscription checkout without a customer or with `subscription_data`
beyond its metadata (a trial). Subscription schedules (paid
downgrades) aren't modelled at all. That list is never trimmed. A journey
reads `GET /__stripe__/requests/cursor` when it starts and checks that nothing
after that is unserved. A request needs a test key (`sk_test_…`); a live key
is refused. As Stripe does, the fake refuses an inline price (`price_data`)
without a product (`product` or `product_data`) or an amount (`unit_amount`
or `unit_amount_decimal`), a recurring price in payment mode, and a
subscription checkout whose prices recur on different intervals (that takes
flexible billing, which this API version lacks). It keeps no result under an
idempotency key for a request it refused. It totals a line as
Stripe does: the unit price times the quantity, rounded up to a whole minor
unit.

**Checkout.** A session's `url` is
`$FAKE_STRIPE_PUBLIC_URL/stripe/checkout/<id>` (the host port, for the
browser). Paying completes the session the way a card that needs no extra
step does:

- **Pay** (payment mode): a payment intent, the card saved to the customer
  when the session asked to keep it, the session `complete` and `paid`; then
  `checkout.session.completed`.
- **Subscribe** (subscription mode): the card saved to the customer, an
  active subscription with one period (a calendar month or year) and its
  first invoice (`subscription_create`), paid with the card; the session
  `complete` and `paid`, with its `subscription` and `invoice`. Then
  `customer.subscription.created`, `invoice.paid`,
  `invoice.payment_succeeded` and `checkout.session.completed`.

When both `FAKE_STRIPE_WEBHOOK_URL` and `FAKE_STRIPE_WEBHOOK_SECRET` are set,
the fake POSTs each event to that URL, signed with that secret as Stripe
signs (`Stripe-Signature: t=<ts>,v1=<HMAC-SHA256 of "<ts>.<payload>">`), and
only then redirects to `success_url`. So unless a journey held them, the
platform has the webhooks by the time the user is back. If either is unset,
nothing is sent: each delivery is recorded as failed ("no webhook endpoint").
**Cancel** links to `cancel_url`.

**Webhook order.** Events go out one at a time, in the order they happened;
one the platform never answers (refused, timed out, cut short) is recorded as
a failed delivery and the next goes out. One thread sends them, and only one:
`FakeStripe.close()`, which the tests call, stops it once the queue is out,
and an event after that starts a new one.
What the fake's own pages and controls do waits for its events to be
delivered. What the platform asks for through the API doesn't: as with
Stripe, the API answers first and the webhook follows a moment later. A
change to a subscription records its events in the same step as the change,
so two changes made at once are announced in the order they were made.

**Billing portal.** A portal session's `url` is
`$FAKE_STRIPE_PUBLIC_URL/stripe/billing_portal/<id>`. It lists the customer's
plans that haven't ended, with their price and when they renew. **Cancel
plan** sets one to cancel at the end of its period, as Stripe's portal does in
that mode, and **Renew plan** takes it back; each sends
`customer.subscription.updated` before the page answers. **Back** returns to
the session's `return_url`. An invoice's `hosted_invoice_url` is a page on the
fake too.

**Webhook controls** (`frontend/src/playwright/journeys/fake-stripe.ts` wraps
them):

- `POST /__stripe__/hold?object=<id>`: keep the events the object's changes
  raise (paying for a session) until the journey delivers them, as when the
  webhook reaches the platform after the user is back. Holding a checkout
  holds the events of the subscription and invoice paying it starts.
- `GET /__stripe__/events?object=<id>`: the events about an object, each with
  every delivery and the platform's status and body; `?customer=<id>` lists
  those about the customer's objects.
- `POST /__stripe__/events/<event id>/deliver`: send an event again, as Stripe
  retries one (same id and body, new signature), or a held one at last.
- `POST /__stripe__/events` with `{"type": ..., "object": "<id>"}`: a new
  event about the object as it is now, e.g.
  `checkout.session.async_payment_succeeded` for a paid session.

**Clock and objects:**

- `POST /__stripe__/subscriptions/<id>/end_period`: end an active
  subscription's current period now, as if the clock had reached it. One set
  to cancel at the period's end is cancelled
  (`customer.subscription.deleted`); any other renews for a new period with a
  new paid invoice (`subscription_cycle`; `customer.subscription.updated`,
  `invoice.paid`, `invoice.payment_succeeded`). It answers with the
  subscription and those events once they are delivered.
- `GET /__stripe__/objects/<id>`: an object as the API shows it, so a journey
  reads what Stripe holds without a key.

## Run it

With the stack, add the override file:

```bash
cd autogpt_platform
docker compose -f docker-compose.yml -f docker-compose.journeys.yml up -d
```

On its own (standard library only, Python 3.10+):

```bash
cd autogpt_platform/scripted_model
SCRIPTED_MODEL_SCRIPTS_DIR=../frontend/src/playwright/journeys/model-scripts \
  SCRIPTED_MODEL_PORT=4010 python -m scripted_model
```

The fake Stripe sends webhooks only when `FAKE_STRIPE_WEBHOOK_URL` and
`FAKE_STRIPE_WEBHOOK_SECRET` are set; `FAKE_STRIPE_PUBLIC_URL` (default
`http://localhost:4010`) is where the browser reaches its pages.

Tests, with the backend's environment (they use the real `anthropic` and
`openai` clients):

```bash
cd autogpt_platform/backend
poetry run pytest ../scripted_model/tests
```
