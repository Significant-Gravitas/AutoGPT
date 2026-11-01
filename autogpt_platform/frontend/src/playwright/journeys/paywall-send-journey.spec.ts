import type { Page } from "@playwright/test";
import { CopilotPage } from "../pages/copilot.page";
import { expect, test } from "./fixtures";

const SUBSCRIPTION_PATH = "/api/proxy/api/credits/subscription";

test.use({
  // The backend's own paywall answers this send: a NO_TIER account with
  // `enable-platform-payment` on gets a 402 from session creation.
  expectedResponses: [
    { status: 402, url: "/api/proxy/api/chat/sessions", method: "POST" },
  ],
});

test("a send after the plan lapses offers one way to upgrade and keeps the message", async ({
  page,
  journeyUser: _journeyUser,
}) => {
  test.skip(
    Boolean(process.env.PLAYWRIGHT_JOURNEY_EMAIL),
    "cancels the user's plan, so it never runs as a real account",
  );
  expect(
    await currentTier(page),
    "the journey user starts on a plan (PLAYWRIGHT_JOURNEY_TIER, e.g. PRO)",
  ).not.toBe("NO_TIER");

  const copilot = new CopilotPage(page);
  // Sending before the AI connections load is refused client-side ("AI
  // connections are still loading"), which is not what this journey is about.
  const connectionsLoaded = page.waitForResponse(
    (response) =>
      response.url().includes("/api/chat/transports") && response.ok(),
  );
  await copilot.open();
  await connectionsLoaded;
  const composer = page.locator("#chat-input-empty");
  await expect(composer).toBeEditable();

  // The plan ends while the page is open: a cancelled plan or a lapsed trial
  // drops the account to NO_TIER, and nothing tells the open page.
  await cancelPlan(page);

  const message = `paywall journey ${Date.now()}`;
  await composer.fill(message);
  await composer.press("Enter");

  const toasts = page.locator("[data-sonner-toast]");
  const upgrade = toasts.getByRole("link", { name: "See plans" });
  await expect(upgrade).toBeVisible();
  await expect(
    toasts,
    "a paywalled send must show one message, not a stack of contradictory ones",
  ).toHaveCount(1);
  await expect(page.getByText("Please try again")).toHaveCount(0);
  await expect(
    composer,
    "the unsent message stays in the composer",
  ).toHaveValue(message);

  await upgrade.click();
  await expect(page).toHaveURL(/\/settings\/billing/);
  await expect(page.getByRole("heading", { name: "Billing" })).toBeVisible();
  await expect(toasts).toHaveCount(0);
});

async function currentTier(page: Page) {
  const status = await page.request.get(SUBSCRIPTION_PATH);
  expect(status.ok(), `reading the plan failed: ${status.status()}`).toBe(true);
  return (await status.json()).tier as string;
}

// A plan granted without Stripe (no customer) is downgraded in the database at
// once, the same state a lapsed trial or an ended subscription leaves.
async function cancelPlan(page: Page) {
  const cancel = await page.request.post(SUBSCRIPTION_PATH, {
    data: { tier: "NO_TIER" },
  });
  expect(
    cancel.ok(),
    `cancelling the plan failed: ${cancel.status()} ${await cancel.text()}`,
  ).toBe(true);
  expect(await currentTier(page)).toBe("NO_TIER");
}
