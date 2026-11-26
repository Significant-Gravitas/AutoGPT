import type { Page } from "@playwright/test";
import {
  checkoutSessionID,
  customerStripeEvents,
  endSubscriptionPeriod,
  requireFakeStripe,
  type StripeInvoice,
  stripeObject,
  stripeRequestCursor,
  type StripeSubscription,
  unservedStripeRequests,
} from "./fake-stripe";
import { expect, test } from "./fixtures";

// The Pro plan on Settings → Billing → Subscription, bought on the fake
// Stripe's checkout page and cancelled in its billing portal, as a customer
// does both (fake-stripe.ts), or through the platform's API. The fake's prices
// (scripted_model/scripted_model/fake_stripe/catalog.json) differ from the
// frontend's marketing prices, so a price on the page is the one Stripe charges.
const PRO_PRICE = "$23.00";
const PRO_PRICE_ID = "price_journeys_pro_monthly";
const SUBSCRIPTION_PATH = "/api/proxy/api/credits/subscription";
// Starting a chat is behind the paywall: without a plan it is refused (402).
const CHAT_SESSIONS_PATH = "/api/proxy/api/chat/sessions";
const NO_PLAN = {
  label: "No active subscription",
  badge: "Inactive",
  summary: "Pick a plan to continue using AutoGPT.",
};
const SUBSCRIBED = [
  "customer.subscription.created",
  "invoice.paid",
  "invoice.payment_succeeded",
  "checkout.session.completed",
];

// A new hosted user has no plan; this journey buys one.
test.use({ grantJourneyTier: false });

test.describe("subscription journey", () => {
  test("a plan bought at checkout and cancelled in the billing portal lasts until its period ends", async ({
    page,
    journeyUser: _user,
  }) => {
    await requireNewAccountAndFakeStripe();
    const stripeCursor = await stripeRequestCursor();
    await expectPaywalled(page, true);

    const { customer, subscription } = await buyPro(page);
    await expectPaywalled(page, false);

    await cancelInBillingPortal(page);
    // Pro until the period ends, and the page says when that is.
    const ends = await shortDate(page, subscription.current_period_end);
    await expectPlan(page, {
      label: "Pro",
      badge: "Cancellation scheduled",
      summary: `${PRO_PRICE} / month · Ends on ${ends}`,
    });
    await expect(
      planCard(page, "Pro").getByRole("button", {
        name: "Resume subscription",
      }),
    ).toBeVisible();
    expect(await planStatus(page)).toMatchObject({
      tier: "PRO",
      pending_tier: "NO_TIER",
    });
    await expectPaywalled(page, false);

    const ended = await endSubscriptionPeriod(subscription.id);
    expect(ended.subscription.status).toBe("canceled");
    await page.reload();
    await expectPlan(page, NO_PLAN);
    expect(await planStatus(page)).toMatchObject({
      tier: "NO_TIER",
      has_active_stripe_subscription: false,
      pending_tier: null,
    });
    await expectPaywalled(page, true);

    await expectWebhooksAcknowledged(customer, [
      ...SUBSCRIBED,
      "customer.subscription.updated",
      "customer.subscription.deleted",
    ]);
    expect(await unservedStripeRequests(stripeCursor)).toEqual([]);
  });

  test("a cancelled plan resumed before its period ends renews", async ({
    page,
    journeyUser: _user,
  }) => {
    await requireNewAccountAndFakeStripe();
    const stripeCursor = await stripeRequestCursor();
    const { customer, subscription } = await buyPro(page);
    await cancelInBillingPortal(page);
    const periodEnd = await shortDate(page, subscription.current_period_end);
    await expectPlan(page, {
      label: "Pro",
      badge: "Cancellation scheduled",
      summary: `${PRO_PRICE} / month · Ends on ${periodEnd}`,
    });

    const resume = planCard(page, "Pro").getByRole("button", {
      name: "Resume subscription",
    });
    await expect(resume).toBeVisible();
    await resume.click();
    await expect(page.getByText("Subscription resumed")).toBeVisible();
    await expectPlan(page, {
      label: "Pro",
      badge: "Active",
      summary: `${PRO_PRICE} / month · Renews on ${periodEnd}`,
    });
    const resumed = await stripeObject<StripeSubscription>(subscription.id);
    expect(resumed.cancel_at_period_end, "resumed in Stripe too").toBe(false);

    const renewed = await endSubscriptionPeriod(subscription.id);
    expect(renewed.subscription.status).toBe("active");
    await page.reload();
    const next = renewed.subscription.current_period_end;
    await expectPlan(page, {
      label: "Pro",
      badge: "Active",
      summary: `${PRO_PRICE} / month · Renews on ${await shortDate(page, next)}`,
    });
    expect(await planStatus(page)).toMatchObject({
      tier: "PRO",
      pending_tier: null,
      current_period_end: next,
    });
    await expectPaywalled(page, false);
    // Both payments are on the invoices list.
    for (const invoiceID of [
      subscription.latest_invoice,
      renewed.subscription.latest_invoice,
    ]) {
      const invoice = await stripeObject<StripeInvoice>(String(invoiceID));
      const row = page.getByRole("row").filter({ hasText: invoice.number });
      await expect(row).toContainText(PRO_PRICE);
      await expect(row).toContainText("paid");
    }

    await expectWebhooksAcknowledged(customer, [
      ...SUBSCRIBED,
      // cancelled in the portal, resumed in the app, renewed
      "customer.subscription.updated",
      "customer.subscription.updated",
      "customer.subscription.updated",
      "invoice.paid",
      "invoice.payment_succeeded",
    ]);
    expect(await unservedStripeRequests(stripeCursor)).toEqual([]);
  });

  test("a plan cancelled through the platform's API lasts until its period ends", async ({
    page,
    journeyUser: _user,
  }) => {
    await requireNewAccountAndFakeStripe();
    const stripeCursor = await stripeRequestCursor();
    const { customer, subscription } = await buyPro(page);

    // The Pro plan card has no cancel of its own: "Manage subscription"
    // opens Stripe's billing portal (above). So this sends the cancel the
    // platform's API takes, a change to no plan, from the signed-in page.
    // The platform sets the subscription to cancel when its period ends.
    expect(await changePlan(page, "NO_TIER")).toMatchObject({
      tier: "PRO",
      pending_tier: "NO_TIER",
      current_period_end: subscription.current_period_end,
    });
    const inStripe = await stripeObject<StripeSubscription>(subscription.id);
    expect(inStripe.cancel_at_period_end, "set to cancel in Stripe").toBe(true);
    await page.reload();
    const periodEnd = await shortDate(page, subscription.current_period_end);
    await expectPlan(page, {
      label: "Pro",
      badge: "Cancellation scheduled",
      summary: `${PRO_PRICE} / month · Ends on ${periodEnd}`,
    });
    await expectPaywalled(page, false);

    const ended = await endSubscriptionPeriod(subscription.id);
    expect(ended.subscription.status).toBe("canceled");
    await page.reload();
    await expectPlan(page, NO_PLAN);
    expect(await planStatus(page)).toMatchObject({
      tier: "NO_TIER",
      has_active_stripe_subscription: false,
      pending_tier: null,
    });
    await expectPaywalled(page, true);

    await expectWebhooksAcknowledged(customer, [
      ...SUBSCRIBED,
      "customer.subscription.updated",
      "customer.subscription.deleted",
    ]);
    expect(await unservedStripeRequests(stripeCursor)).toEqual([]);
  });
});

async function requireNewAccountAndFakeStripe() {
  test.skip(
    Boolean(process.env.PLAYWRIGHT_JOURNEY_EMAIL),
    "buys and cancels a plan, so it never runs as a real account",
  );
  await requireFakeStripe();
}

/**
 * Buys Pro from Billing → Subscription with no plan yet, on the fake's
 * checkout page, and checks the user is back on it.
 */
async function buyPro(page: Page) {
  await page.goto("/settings/billing?tab=subscription");
  await expectPlan(page, NO_PLAN);
  // The plans, at Stripe's prices.
  const offer = page.getByRole("region", { name: "AutoGPT Pro" });
  await expect(offer.getByText(PRO_PRICE, { exact: true })).toBeVisible();
  await expect(offer.getByText("/ month", { exact: true })).toBeVisible();
  expect(await planStatus(page)).toMatchObject({
    tier: "NO_TIER",
    tier_costs: { PRO: 2300, MAX: 9700 },
    tier_costs_yearly: { PRO: 23000, MAX: 97000 },
  });

  await offer.getByRole("button", { name: "Start Pro" }).click();
  await expect(
    page,
    "checkout should open on the fake Stripe (STRIPE_API_BASE)",
  ).toHaveURL(/\/stripe\/checkout\/cs_test_/);
  const sessionID = checkoutSessionID(page.url());
  await expect(
    page.getByText(`AutoGPT Pro × 1: ${PRO_PRICE} / month`),
  ).toBeVisible();
  await expect(page.getByText(`Total: ${PRO_PRICE}`)).toBeVisible();
  // Stripe's webhooks reach the platform before the user is back.
  await page.getByRole("button", { name: "Subscribe", exact: true }).click();
  await expect(page).toHaveURL(/\/settings\/billing/, { timeout: 30_000 });
  // Back from checkout, the app tells the user the plan is on.
  const activated = page
    .getByRole("dialog")
    .filter({ hasText: "Pro is active" });
  await expect(activated).toContainText("Your Pro plan is active.");
  await activated
    .getByRole("button", { name: "Continue where you left off" })
    .click();
  await expect(activated).toBeHidden();

  const session = await stripeObject<{
    customer: string;
    subscription: string;
  }>(sessionID);
  const subscription = await stripeObject<StripeSubscription>(
    session.subscription,
  );
  expect(subscription.items.data.map((item) => item.price.id)).toEqual([
    PRO_PRICE_ID,
  ]);
  // On the plan, as the page shows it and as Stripe has it.
  const renews = await shortDate(page, subscription.current_period_end);
  await expectPlan(page, {
    label: "Pro",
    badge: "Active",
    summary: `${PRO_PRICE} / month · Renews on ${renews}`,
  });
  expect(await planStatus(page)).toMatchObject({
    tier: "PRO",
    monthly_cost: 2300,
    has_active_stripe_subscription: true,
    current_period_end: subscription.current_period_end,
  });
  return { customer: session.customer, subscription };
}

/** Cancels Pro in the billing portal, as Stripe's portal does it, and goes back. */
async function cancelInBillingPortal(page: Page) {
  const manage = planCard(page, "Pro").getByRole("button", {
    name: "Manage subscription",
  });
  await expect(manage).toBeEnabled();
  await manage.click();
  await expect(
    page,
    "the billing portal should be the fake Stripe's",
  ).toHaveURL(/\/stripe\/billing_portal\/bps_/);
  const plan = page.getByRole("region", { name: "AutoGPT Pro" });
  await expect(plan).toContainText(`${PRO_PRICE} per month`);
  // Its webhook reaches the platform before the portal answers.
  await plan.getByRole("button", { name: "Cancel plan" }).click();
  await expect(plan).toContainText("Your plan will be canceled on");
  await page.getByRole("link", { name: "Back" }).click();
  await expect(page).toHaveURL(/\/settings\/billing/);
}

interface PlanView {
  label: string;
  badge: string;
  summary: string;
}

async function expectPlan(page: Page, { label, badge, summary }: PlanView) {
  const card = planCard(page, label);
  await expect(card.getByText(badge, { exact: true })).toBeVisible();
  await expect(card.getByText(summary, { exact: true })).toBeVisible();
}

/** The "Your plan" card, while it shows the plan labelled `label`. */
function planCard(page: Page, label: string) {
  return page.locator("section").filter({
    has: page.getByRole("heading", { level: 2, name: label, exact: true }),
  });
}

/** Asks the platform for a plan change, as the billing page's buttons do. */
async function changePlan(page: Page, tier: string) {
  const response = await page.request.post(SUBSCRIPTION_PATH, {
    data: { tier },
  });
  expect(
    response.ok(),
    `changing the plan to ${tier}: ${response.status()} ${await response.text()}`,
  ).toBe(true);
  return (await response.json()) as Record<string, unknown>;
}

async function planStatus(page: Page) {
  const response = await page.request.get(SUBSCRIPTION_PATH);
  expect(response.ok(), `GET subscription: ${response.status()}`).toBe(true);
  return (await response.json()) as Record<string, unknown>;
}

async function expectPaywalled(page: Page, paywalled: boolean) {
  const response = await page.request.post(CHAT_SESSIONS_PATH, { data: {} });
  expect(
    response.status(),
    `starting a chat ${paywalled ? "without" : "on"} a plan: ${await response.text()}`,
  ).toBe(paywalled ? 402 : 200);
}

/** A Stripe time as the page writes dates (`formatShortDate`), in its locale. */
function shortDate(page: Page, unixSeconds: number) {
  return page.evaluate(
    (ms) =>
      new Date(ms).toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        year: "numeric",
      }),
    unixSeconds * 1000,
  );
}

/**
 * Every webhook the fake sent about the customer's objects, in order, each
 * delivered once and acknowledged. One the platform's own API call raised
 * arrives after that call's answer, so wait for it.
 */
async function expectWebhooksAcknowledged(customerID: string, types: string[]) {
  await expect
    .poll(
      async () =>
        (await customerStripeEvents(customerID)).filter(
          (event) => event.deliveries.length > 0,
        ).length,
      { message: "every webhook should have been delivered" },
    )
    .toBeGreaterThanOrEqual(types.length);
  const events = await customerStripeEvents(customerID);
  expect(
    events.map((event) => ({
      type: event.type,
      statuses: event.deliveries.map((delivery) => delivery.status),
    })),
    "the platform should acknowledge each webhook",
  ).toEqual(types.map((type) => ({ type, statuses: [200] })));
}
