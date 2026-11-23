import { expect, test } from "@playwright/test";
import { isLocalStack } from "./journey-user";

/**
 * The hosted journeys stack's Stripe is a fake in the scripted_model service
 * (autogpt_platform/scripted_model/README.md, "Fake Stripe"): the platform's
 * Stripe SDK calls it, the browser pays on its checkout page, and it signs
 * and sends the webhooks. These read and drive its webhook deliveries.
 */
const FAKE_STRIPE_URL =
  process.env.PLAYWRIGHT_FAKE_STRIPE_URL ?? "http://localhost:4010";

export interface StripeDelivery {
  /** The platform's HTTP status, or null when it never answered. */
  status: number | null;
  body: string;
  error: string | null;
}

export interface StripeEvent {
  id: string;
  type: string;
  /** The id of the object the event is about, e.g. the checkout session. */
  object: string;
  /** The customer whose object it is. */
  customer: string | null;
  held: boolean;
  event: {
    data: {
      object: Record<string, unknown>;
      /** What an `*.updated` event's object had before the change. */
      previous_attributes?: Record<string, unknown>;
    };
  };
  deliveries: StripeDelivery[];
}

/** A subscription as the fake Stripe holds it, the fields journeys read. */
export interface StripeSubscription {
  id: string;
  customer: string;
  status: string;
  cancel_at_period_end: boolean;
  current_period_start: number;
  current_period_end: number;
  latest_invoice: string | null;
  items: { data: { price: { id: string; unit_amount: number } }[] };
}

export interface StripeInvoice {
  id: string;
  number: string;
  status: string;
  total: number;
  billing_reason: string;
}

/**
 * Skips the journey against a deployed app, which talks to Stripe itself.
 * On a local stack, fails unless the fake Stripe is up.
 */
export async function requireFakeStripe() {
  test.skip(
    !isLocalStack(),
    "the fake Stripe runs only in a local stack with docker-compose.journeys.yml",
  );
  const up = await fetch(`${FAKE_STRIPE_URL}/__stripe__/requests`)
    .then((response) => response.ok)
    .catch(() => false);
  if (!up) {
    throw new Error(
      `the fake Stripe isn't answering at ${FAKE_STRIPE_URL}. Start the stack ` +
        "with docker-compose.journeys.yml and the hosted backend settings " +
        "(see src/playwright/journeys/README.md).",
    );
  }
}

/** The checkout session a fake checkout page is for. */
export function checkoutSessionID(pageURL: string) {
  const match = /\/stripe\/checkout\/(cs_test_[A-Za-z0-9]+)$/.exec(
    new URL(pageURL).pathname,
  );
  if (!match) throw new Error(`${pageURL} is not a fake Stripe checkout page`);
  return match[1];
}

/**
 * Keeps the webhooks paying for `objectID` raises until the journey
 * delivers them, as when Stripe's webhook reaches the platform after the
 * user is back on its page.
 */
export async function holdWebhooks(objectID: string) {
  await control<{ held: string }>(
    "POST",
    `hold?object=${encodeURIComponent(objectID)}`,
  );
}

/** Every webhook event about `objectID`, oldest first, with its deliveries. */
export async function stripeEvents(objectID: string): Promise<StripeEvent[]> {
  const body = await control<{ events: StripeEvent[] }>(
    "GET",
    `events?object=${encodeURIComponent(objectID)}`,
  );
  return body.events;
}

/** Every webhook event about the customer's objects, oldest first. */
export async function customerStripeEvents(
  customerID: string,
): Promise<StripeEvent[]> {
  const body = await control<{ events: StripeEvent[] }>(
    "GET",
    `events?customer=${encodeURIComponent(customerID)}`,
  );
  return body.events;
}

/**
 * An object as the fake Stripe holds it, as its API shows it: what Stripe
 * says about a subscription or invoice, to hold the app's page against.
 */
export async function stripeObject<T>(objectID: string): Promise<T> {
  return control<T>("GET", `objects/${encodeURIComponent(objectID)}`);
}

/**
 * Ends a subscription's current period now, as if the clock had reached it.
 * One set to cancel at the period's end is cancelled; any other renews with a
 * new paid invoice. Returns once the platform has had the webhooks.
 */
export async function endSubscriptionPeriod(subscriptionID: string) {
  return control<{ subscription: StripeSubscription; events: StripeEvent[] }>(
    "POST",
    `subscriptions/${encodeURIComponent(subscriptionID)}/end_period`,
  );
}

/** Delivers an event again, as Stripe retries one, or a held one at last. */
export async function deliverEvent(eventID: string): Promise<StripeDelivery> {
  const body = await control<{ delivery: StripeDelivery }>(
    "POST",
    `events/${encodeURIComponent(eventID)}/deliver`,
  );
  return body.delivery;
}

/** Sends a new event of `type` about `objectID` as it is now. */
export async function sendEvent(
  type: string,
  objectID: string,
): Promise<StripeEvent> {
  return control<StripeEvent>("POST", "events", { type, object: objectID });
}

/**
 * Where the fake's request journal is now. Take it at a journey's start, and
 * give it to `unservedStripeRequests` to look only at the journey's calls.
 */
export async function stripeRequestCursor(): Promise<number> {
  const body = await control<{ cursor: number }>("GET", "requests/cursor");
  return body.cursor;
}

/**
 * Stripe API calls the platform made after `cursor` that the fake doesn't
 * model.
 */
export async function unservedStripeRequests(
  cursor: number,
): Promise<StripeRequest[]> {
  const body = await control<{ requests: StripeRequest[] }>(
    "GET",
    `requests?unserved=true&after=${cursor}`,
  );
  return body.requests;
}

/** A Stripe API call the platform made, as the fake journals it. */
export interface StripeRequest {
  seq: number;
  method: string;
  path: string;
  status: number;
  served: boolean;
}

async function control<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const response = await fetch(`${FAKE_STRIPE_URL}/__stripe__/${path}`, {
    method,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(30_000),
  });
  expect(
    response.ok,
    `${method} fake Stripe ${path}: ${response.status} ${await response.clone().text()}`,
  ).toBe(true);
  return (await response.json()) as T;
}
