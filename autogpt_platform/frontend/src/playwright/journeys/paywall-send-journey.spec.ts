import type { Page } from "@playwright/test";
import { CopilotPage } from "../pages/copilot.page";
import { expect, test } from "./fixtures";

// PostHog's edge serves no flags to a "HeadlessChrome" user agent, so a
// headless run would not see the product hosted users see.
const CHROME_USER_AGENT =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36";

const CREATE_SESSION_PATH = "/api/proxy/api/chat/sessions";
// What the hosted backend answers a NO_TIER account (backend/api/rest_api.py
// maps UserPaywalledError to this).
const PAYWALL_DETAIL =
  "A subscription is required to use this feature. Upgrade to continue.";

test.use({
  userAgent: CHROME_USER_AGENT,
  expectedResponses: [
    { status: 402, url: CREATE_SESSION_PATH, method: "POST" },
  ],
});

test("a send after the plan lapses offers one way to upgrade and keeps the message", async ({
  page,
  journeyUser: _journeyUser,
}) => {
  const copilot = new CopilotPage(page);
  await copilot.open();
  const composer = page.locator("#chat-input-empty");
  await expect(composer).toBeEditable();

  // The plan ends while the page is open: a cancelled plan or a lapsed trial
  // drops the account to NO_TIER, and nothing tells the open page.
  await cancelPlan(page);
  await answerSendsLikeHostedPaywall(page);

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

async function cancelPlan(page: Page) {
  const cancel = await page.request.post(
    "/api/proxy/api/credits/subscription",
    { data: { tier: "NO_TIER" } },
  );
  expect(
    cancel.ok(),
    `cancelling the plan failed: ${cancel.status()} ${await cancel.text()}`,
  ).toBe(true);
  const status = await page.request.get("/api/proxy/api/credits/subscription");
  expect((await status.json()).tier).toBe("NO_TIER");
}

// The journeys leg forces `enable-platform-payment` off on the backend so
// fresh users elsewhere aren't paywalled (journeys/README.md). Where the
// backend already paywalls (hosted config, previews) its own 402 passes
// through untouched; otherwise the session it created is answered the way
// the hosted backend answers this NO_TIER account.
async function answerSendsLikeHostedPaywall(page: Page) {
  await page.route(`**${CREATE_SESSION_PATH}`, async (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    const response = await route.fetch();
    if (response.status() === 402) return route.fulfill({ response });
    return route.fulfill({ status: 402, json: { detail: PAYWALL_DETAIL } });
  });
}
