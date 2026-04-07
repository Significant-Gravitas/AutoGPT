import { expect, test } from "./coverage-fixture";
import { getSeededTestUser } from "./credentials/accounts";
import { LoginPage } from "./pages/login.page";

async function dismissCopilotNotificationPrompt(
  page: import("@playwright/test").Page,
) {
  const notNowButton = page.getByRole("button", { name: "Not now" });
  if (await notNowButton.isVisible({ timeout: 3000 }).catch(() => false)) {
    await notNowButton.click();
  }
}

test("copilot happy path: user can create a deterministic AutoPilot session and keep it after reload", async ({
  page,
}) => {
  test.setTimeout(120000);

  const loginPage = new LoginPage(page);
  const copilotUser = getSeededTestUser("smokeMarketplace");

  await page.goto("/login");
  await loginPage.login(copilotUser.email, copilotUser.password);

  await page.goto("/copilot");
  await expect(page).toHaveURL(/\/copilot/);
  await dismissCopilotNotificationPrompt(page);

  const response = await page.request.post("/api/proxy/api/chat/sessions", {
    data: null,
  });
  expect(response.ok()).toBeTruthy();

  const session = await response.json();
  const sessionId = session?.id;
  expect(sessionId).toBeTruthy();

  await page.goto(`/copilot?sessionId=${sessionId}`);
  await dismissCopilotNotificationPrompt(page);
  await expect(page.locator("#chat-input-session")).toBeVisible({
    timeout: 15000,
  });

  await page.reload();
  await page.waitForLoadState("domcontentloaded");
  await dismissCopilotNotificationPrompt(page);

  await expect
    .poll(() => new URL(page.url()).searchParams.get("sessionId"), {
      timeout: 15000,
    })
    .toBe(sessionId);
  await expect(page.locator("#chat-input-session")).toBeVisible({
    timeout: 15000,
  });
});
