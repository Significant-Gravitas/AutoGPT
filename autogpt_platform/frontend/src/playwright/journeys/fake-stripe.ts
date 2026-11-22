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
  held: boolean;
  event: { data: { object: Record<string, unknown> } };
  deliveries: StripeDelivery[];
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
  await control("POST", `hold?object=${encodeURIComponent(objectID)}`);
}

/** Every webhook event about `objectID`, oldest first, with its deliveries. */
export async function stripeEvents(objectID: string): Promise<StripeEvent[]> {
  const body = await control(
    "GET",
    `events?object=${encodeURIComponent(objectID)}`,
  );
  return body.events;
}

/** Delivers an event again, as Stripe retries one, or a held one at last. */
export async function deliverEvent(eventID: string): Promise<StripeDelivery> {
  const body = await control(
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
  return control("POST", "events", { type, object: objectID });
}

/** Stripe API calls the platform made that the fake doesn't model. */
export async function unservedStripeRequests(): Promise<
  { method: string; path: string; status: number }[]
> {
  const body = await control("GET", "requests?unserved=true");
  return body.requests;
}

async function control(method: string, path: string, body?: unknown) {
  const response = await fetch(`${FAKE_STRIPE_URL}/__stripe__/${path}`, {
    method,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  expect(
    response.ok,
    `${method} fake Stripe ${path}: ${response.status} ${await response.clone().text()}`,
  ).toBe(true);
  return response.json();
}
