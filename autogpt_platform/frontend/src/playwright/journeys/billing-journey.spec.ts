import type { Page, Response } from "@playwright/test";
import { getCreditBalance, getCreditTransactions } from "./credits";
import {
  checkoutSessionID,
  deliverEvent,
  holdWebhooks,
  requireFakeStripe,
  sendEvent,
  stripeEvents,
  unservedStripeRequests,
} from "./fake-stripe";
import { expect, test } from "./fixtures";

// Buying credits on Settings → Billing, paid on the fake Stripe's checkout
// page (fake-stripe.ts). One credit is one cent, so $7 buys 700.
const TOP_UP_DOLLARS = 7;
const TOP_UP = TOP_UP_DOLLARS * 100;
const CREDITS_PATH = "/api/proxy/api/credits";
const USD = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
});

test.describe("billing journey", () => {
  test(
    "credits bought through Stripe checkout are added once, however often Stripe reports the payment",
    {
      annotation: {
        type: "issue",
        description:
          "https://github.com/Significant-Gravitas/AutoGPT/issues/17270",
      },
    },
    async ({ page, journeyUser: _user }) => {
      await requireFakeStripe();
      const before = await getCreditBalance(page);
      await openAutomationCredits(page);
      await expect(balance(page, before)).toBeVisible();

      const sessionID = await checkOut(page);
      await payAndReturn(page);

      // The fake sends checkout.session.completed before it sends the user back.
      const events = await stripeEvents(sessionID);
      expect(events.map((event) => event.type)).toEqual([
        "checkout.session.completed",
      ]);
      const [completed] = events;
      expect(
        completed.deliveries.map((delivery) => delivery.status),
        "the platform should acknowledge the payment",
      ).toEqual([200]);
      await expectAddedOnce(page, before, paymentIntent(completed));

      // Stripe sends an event again when it doesn't hear back in time.
      const retried = await deliverEvent(completed.id);
      expect(retried.status, retried.body).toBe(200);
      // A delayed payment method ends with a second event about the session.
      const succeeded = await sendEvent(
        "checkout.session.async_payment_succeeded",
        sessionID,
      );
      expect(succeeded.deliveries.map((delivery) => delivery.status)).toEqual([
        200,
      ]);
      // And the return page asks the platform to fulfil it on every visit.
      const fulfil = await page.request.patch(CREDITS_PATH);
      expect(fulfil.ok(), `PATCH credits: ${fulfil.status()}`).toBe(true);

      // The toast says to refresh if the credits don't show; the user does.
      await page.reload();
      const creditsTab = page.getByRole("tab", { name: "Automation Credits" });
      await expect(creditsTab).toBeVisible();
      const reloadedOnCredits =
        (await creditsTab.getAttribute("aria-selected")) === "true";
      await creditsTab.click();
      await expectAddedOnce(page, before, paymentIntent(completed));
      expect(await unservedStripeRequests()).toEqual([]);

      // A known failure from here on, so anything above failing still fails
      // the run.
      test.fail(
        true,
        "#17270: a reload after buying credits opens the Subscription tab",
      );
      expect(
        reloadedOnCredits,
        "the reload should stay on Automation Credits, where the toast said the credits would appear",
      ).toBe(true);
    },
  );

  test(
    "credits whose webhook reaches the platform after the user is back are added once, and the page shows them",
    {
      annotation: {
        type: "issue",
        description:
          "https://github.com/Significant-Gravitas/AutoGPT/issues/17269",
      },
    },
    async ({ page, journeyUser: _user }) => {
      await requireFakeStripe();
      const before = await getCreditBalance(page);
      await openAutomationCredits(page);
      const sessionID = await checkOut(page);
      await holdWebhooks(sessionID);

      // With the webhook late, the return page fulfils the checkout itself.
      // Against Stripe its balance read answers before that call, which waits
      // on Stripe; the fake answers at once, so hold the call until the read
      // is back, whatever the stack's speed.
      const balanceRead = page.waitForResponse(isBalanceRead).catch(() => null);
      await page.route(`**${CREDITS_PATH}`, async (route) => {
        if (route.request().method() === "PATCH") await balanceRead;
        await route.fallback();
      });
      const fulfilled = page.waitForResponse(isFulfilment);
      await payAndReturn(page);
      expect((await fulfilled).ok(), "PATCH credits from the return page").toBe(
        true,
      );
      const [completed, ...others] = await stripeEvents(sessionID);
      expect(others).toEqual([]);
      expect(completed.deliveries, "the webhook is held").toEqual([]);
      await expectInLedgerOnce(page, before, paymentIntent(completed));

      // Then Stripe's webhook arrives.
      const late = await deliverEvent(completed.id);
      expect(late.status, late.body).toBe(200);
      await expectInLedgerOnce(page, before, paymentIntent(completed));
      expect(await unservedStripeRequests()).toEqual([]);

      // A known failure from here on, so anything above failing still fails
      // the run.
      test.fail(
        true,
        "#17269: the page keeps the old balance after its return from checkout credits the top-up",
      );
      await expect(
        balance(page, before + TOP_UP),
        "the page should show the credits its own fulfil call added",
      ).toBeVisible({ timeout: 5_000 });
      await expect(
        purchases(page),
        "the history should list the purchase without a reload",
      ).toHaveCount(1, { timeout: 5_000 });
    },
  );
});

async function openAutomationCredits(page: Page) {
  await page.goto("/settings/billing?tab=automation-credits");
  await expect(
    page.getByRole("tab", { name: "Automation Credits" }),
  ).toHaveAttribute("aria-selected", "true");
}

/** Buys the top-up in the Add credits dialog, up to the checkout page. */
async function checkOut(page: Page) {
  await addCreditsButton(page).click();
  const dialog = page.getByRole("dialog", { name: "Add credits" });
  await dialog.getByLabel(/^Amount/).fill(String(TOP_UP_DOLLARS));
  const proceed = dialog.getByRole("button", { name: "Continue to checkout" });
  await expect(proceed).toBeEnabled();
  await proceed.click();
  await expect(
    page,
    "checkout should open on the fake Stripe (STRIPE_API_BASE)",
  ).toHaveURL(/\/stripe\/checkout\/cs_test_/);
  await expect(
    page.getByText(`Total: ${USD.format(TOP_UP_DOLLARS)}`),
  ).toBeVisible();
  return checkoutSessionID(page.url());
}

/** Pays on the checkout page, which sends the user back to Billing. */
async function payAndReturn(page: Page) {
  await page.getByRole("button", { name: "Pay", exact: true }).click();
  await expect(page).toHaveURL(/\/settings\/billing/);
  await expect(page.getByText("Payment successful")).toBeVisible();
}

/**
 * The top-up is in the ledger once, as the payment Stripe took, and the
 * page shows it once: the balance and one "Credits added" row.
 */
async function expectAddedOnce(page: Page, before: number, payment: string) {
  await expectInLedgerOnce(page, before, payment);
  await expect(balance(page, before + TOP_UP)).toBeVisible();
  await expect(
    purchases(page),
    "the history should list the purchase once",
  ).toHaveCount(1);
  await expect(purchases(page)).toContainText(`+${USD.format(TOP_UP_DOLLARS)}`);
}

async function expectInLedgerOnce(page: Page, before: number, payment: string) {
  expect(
    await getCreditBalance(page),
    "the balance should rise by the top-up, once",
  ).toBe(before + TOP_UP);
  const topUps = (await getCreditTransactions(page)).filter(
    (transaction) => transaction.transaction_type === "TOP_UP",
  );
  expect(
    topUps.map(({ amount, transaction_key }) => ({ amount, transaction_key })),
    "one top-up in the ledger, keyed by the payment",
  ).toEqual([{ amount: TOP_UP, transaction_key: payment }]);
}

function addCreditsButton(page: Page) {
  return page.getByRole("button", { name: "Add credits", exact: true });
}

/** The balance card's figure, when it reads `cents`. */
function balance(page: Page, cents: number) {
  return page
    .locator("section")
    .filter({ has: addCreditsButton(page) })
    .getByText(USD.format(cents / 100), { exact: true });
}

function purchases(page: Page) {
  return page
    .getByRole("table")
    .getByRole("row")
    .filter({ hasText: "Credits added" });
}

/** The payment intent Stripe took the session's money with. */
function paymentIntent(event: { event: { data: { object: object } } }) {
  const session = event.event.data.object as { payment_intent?: unknown };
  expect(typeof session.payment_intent).toBe("string");
  return String(session.payment_intent);
}

function isFulfilment(response: Response) {
  return (
    response.request().method() === "PATCH" &&
    new URL(response.url()).pathname === CREDITS_PATH
  );
}

function isBalanceRead(response: Response) {
  return (
    response.request().method() === "GET" &&
    new URL(response.url()).pathname === CREDITS_PATH
  );
}
