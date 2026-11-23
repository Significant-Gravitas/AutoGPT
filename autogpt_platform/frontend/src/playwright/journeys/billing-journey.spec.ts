import type { Page, Request, Response } from "@playwright/test";
import { getCreditBalance, getCreditTransactions } from "./credits";
import {
  checkoutSessionID,
  deliverEvent,
  holdWebhooks,
  requireFakeStripe,
  sendEvent,
  stripeEvents,
  stripeRequestCursor,
  type StripeEvent,
  unservedStripeRequests,
} from "./fake-stripe";
import { expect, test } from "./fixtures";

// Buying credits on Settings → Billing, paid on the fake Stripe's checkout
// page (fake-stripe.ts). One credit is one cent, so $7 buys 700.
const TOP_UP_DOLLARS = 7;
const TOP_UP = TOP_UP_DOLLARS * 100;
const CREDITS_PATH = "/api/proxy/api/credits";
const HISTORY_PATH = "/api/proxy/api/credits/transactions";
// Page time the second test runs. The page's backup read comes 10 s after its
// fulfil call (LATE_WEBHOOK_MS in useTopupCheckoutReturn.ts), and the clock is
// paused before that call answers, so the backup is armed on page time the
// test runs: the refresh steps plus the steps for the answers, kept far below.
// Keep BACKUP_READ_MS equal to LATE_WEBHOOK_MS; the test can't import it.
const BACKUP_READ_MS = 10_000;
const PAUSE_JUMP_MS = 1_000;
// One step per poll (every 100 ms): REFRESH_CLOCK_BUDGET_MS lasts its poll's
// 15 s, and ANSWER_RUN_MS about 10 s of the two answer polls it's shared by. A
// runner that stalls before the page handles an answer still gets its timers
// run, and the steps only fire due timers, at most each budget's worth.
const REFRESH_STEP_MS = 10;
const REFRESH_CLOCK_BUDGET_MS = 1_500;
const ANSWER_RUN_MS = 1_000;
const USD = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
});

test.describe("billing journey", () => {
  test("credits bought through Stripe checkout are added once, however often Stripe reports the payment", async ({
    page,
    journeyUser: _user,
  }) => {
    await requireFakeStripe();
    const stripeCursor = await stripeRequestCursor();
    const before = await getCreditBalance(page);
    await openAutomationCredits(page);
    await expect(balance(page, before)).toBeVisible();

    const sessionID = await checkOut(page);
    // The return page also fulfils the checkout itself, so hold that call
    // until the purchase is credited: only the webhook can have done it.
    let releaseFulfilment = () => {};
    const fulfilmentHeld = new Promise<void>((resolve) => {
      releaseFulfilment = resolve;
    });
    await page.route(`**${CREDITS_PATH}`, async (route) => {
      if (route.request().method() === "PATCH") await fulfilmentHeld;
      await route.fallback();
    });
    let completed: StripeEvent;
    try {
      await payAndReturn(page);

      // The fake sends checkout.session.completed before the user is back.
      const events = await stripeEvents(sessionID);
      expect(events.map((event) => event.type)).toEqual([
        "checkout.session.completed",
      ]);
      completed = events[0];
      expect(
        completed.deliveries.map((delivery) => delivery.status),
        "the platform should acknowledge the payment",
      ).toEqual([200]);
      await expectAddedOnce(page, before, paymentIntent(completed));
    } finally {
      releaseFulfilment();
    }
    const fulfilled = await page.waitForResponse(isFulfilment, {
      timeout: 15_000,
    });
    expect(fulfilled.ok(), "PATCH credits from the return page").toBe(true);
    await page.unroute(`**${CREDITS_PATH}`);
    await expectInLedgerOnce(page, before, paymentIntent(completed));

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

    // The toast says to refresh if the credits don't show; the user does,
    // and stays where the credits are (#17270).
    await page.reload();
    await expect(
      page.getByRole("tab", { name: "Automation Credits" }),
      "the reload should stay on Automation Credits, where the toast said the credits would appear",
    ).toHaveAttribute("aria-selected", "true");
    await expectAddedOnce(page, before, paymentIntent(completed));
    expect(await unservedStripeRequests(stripeCursor)).toEqual([]);
  });

  test("credits whose webhook reaches the platform after the user is back are added once, and the page shows them", async ({
    page,
    journeyUser: _user,
  }) => {
    expect(
      PAUSE_JUMP_MS + REFRESH_CLOCK_BUDGET_MS + ANSWER_RUN_MS,
      "the page time this test runs should stay far below the backup read",
    ).toBeLessThanOrEqual(BACKUP_READ_MS / 2);
    await requireFakeStripe();
    const stripeCursor = await stripeRequestCursor();
    const before = await getCreditBalance(page);
    await openAutomationCredits(page);
    const sessionID = await checkOut(page);
    await holdWebhooks(sessionID);

    // With the webhook late, the return page fulfils the checkout itself.
    // Against Stripe its balance read answers before that call, which waits
    // on Stripe; the fake answers at once, so hold the call until the read
    // is back, whatever the stack's speed.
    const balanceRead = page
      .waitForResponse(isBalanceRead, { timeout: 15_000 })
      .catch(() => null);
    // A route handler's error doesn't fail the test, so keep it for the body.
    let pauseError: unknown;
    await page.route(`**${CREDITS_PATH}`, async (route) => {
      try {
        if (route.request().method() === "PATCH") {
          await balanceRead;
          // Freeze the page's clock before its fulfil call answers. The page
          // handles the answer, and arms its 10 s backup read, only as this
          // test runs the clock, so time the test doesn't run never counts.
          await page.clock.pauseAt(Date.now() + PAUSE_JUMP_MS);
        }
      } catch (error) {
        pauseError = error;
      } finally {
        // Even if the pause failed, so no request is left hanging.
        await route.fallback();
      }
    });
    let completed: StripeEvent;
    try {
      const refreshReads = getsAfter(page, isFulfilment);
      const fulfilled = page.waitForResponse(isFulfilment);
      // Once its fulfil call is done, the page reads the credits again
      // (#17269), and a second time 10 s later for a webhook still on its
      // way. On the page's own clock, paused before the call answers, that
      // second read can never stand in for the first.
      await page.clock.install();
      await payAndReturn(page);
      expect((await fulfilled).ok(), "PATCH credits from the return page").toBe(
        true,
      );
      // Without the pause the clock runs in real time, and the backup read
      // could stand in for the refresh this test is about.
      expect(
        pauseError,
        "the page's clock should have paused before the fulfil call answered",
      ).toBeUndefined();
      // The wait's timeout still lets the call go, but without the read
      // first this test proves nothing about the page, so it fails here.
      const read = await balanceRead;
      expect(
        read?.ok(),
        "the return page should have read the balance before its fulfil call",
      ).toBe(true);
      // The page handles the answer on timers, which the paused clock fires
      // only when run. Run it a step at a time while waiting, within
      // REFRESH_CLOCK_BUDGET_MS, far short of the backup read, which
      // therefore can't stand in for the first.
      const refreshStep = clockStepper(page, REFRESH_CLOCK_BUDGET_MS);
      await expect
        .poll(
          async () => {
            const paths = refreshReads.map(pathOf);
            const bothRead = [CREDITS_PATH, HISTORY_PATH].every((path) =>
              paths.includes(path),
            );
            await refreshStep(!bothRead);
            return paths;
          },
          {
            message:
              "the page should read the credits and their history again once its fulfil call answered",
            intervals: [100],
          },
        )
        .toEqual(expect.arrayContaining([CREDITS_PATH, HISTORY_PATH]));
      await Promise.all(
        refreshReads
          .filter((request) =>
            [CREDITS_PATH, HISTORY_PATH].includes(pathOf(request)),
          )
          .map((request) => request.response()),
      );
      // Their answers reach the page on its zero-delay timers, which may be
      // set only after a response is seen here: step the clock while
      // waiting for what the page shows, within ANSWER_RUN_MS in all.
      const answerStep = clockStepper(page, ANSWER_RUN_MS);
      await expect
        .poll(
          async () => {
            const shown = await balance(page, before + TOP_UP).isVisible();
            await answerStep(!shown);
            return shown;
          },
          {
            message:
              "the page should show the credits its own fulfil call added",
            intervals: [100],
          },
        )
        .toBe(true);
      await expect
        .poll(
          async () => {
            const rows = await purchases(page).count();
            await answerStep(rows !== 1);
            return rows;
          },
          {
            message: "the history should list the purchase without a reload",
            intervals: [100],
          },
        )
        .toBe(1);
      await page.clock.resume();

      const [held, ...others] = await stripeEvents(sessionID);
      expect(others).toEqual([]);
      expect(held.deliveries, "the webhook is held").toEqual([]);
      completed = held;
      await expectInLedgerOnce(page, before, paymentIntent(completed));
    } finally {
      await page.unroute(`**${CREDITS_PATH}`);
    }

    // Then Stripe's webhook arrives.
    const late = await deliverEvent(completed.id);
    expect(late.status, late.body).toBe(200);
    await expectInLedgerOnce(page, before, paymentIntent(completed));
    expect(await unservedStripeRequests(stripeCursor)).toEqual([]);
  });
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

/**
 * Runs the page's paused clock one REFRESH_STEP_MS step per call while the
 * caller is still waiting, and never more than `budgetMs` in all, so a poll
 * can fire the page's timers without carrying it toward a later one.
 */
function clockStepper(page: Page, budgetMs: number) {
  let run = 0;
  return async (waiting: boolean) => {
    if (!waiting || run >= budgetMs) return;
    await page.clock.runFor(REFRESH_STEP_MS);
    run += REFRESH_STEP_MS;
  };
}

/**
 * Every GET the page sends once a response `isAnswer` accepts has arrived,
 * in order. Both listeners see events in the order the page made them.
 */
function getsAfter(page: Page, isAnswer: (response: Response) => boolean) {
  let answered = false;
  const requests: Request[] = [];
  page.on("response", (response) => {
    if (isAnswer(response)) answered = true;
  });
  page.on("request", (request) => {
    if (answered && request.method() === "GET") requests.push(request);
  });
  return requests;
}

function pathOf(request: Request) {
  return new URL(request.url()).pathname;
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
