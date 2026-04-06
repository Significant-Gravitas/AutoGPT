import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";

test.use({ storageState: E2E_AUTH_STATES.settings });

test("copilot happy path: user can create a deterministic AutoPilot session and keep it after reload", async ({
  page,
}) => {
  test.setTimeout(120000);

  await page.goto("/copilot");
  await expect(page.getByText("Tell me about your work")).toBeVisible({
    timeout: 15000,
  });

  const notNowButton = page.getByRole("button", { name: "Not now" });
  if (await notNowButton.isVisible({ timeout: 3000 }).catch(() => false)) {
    await notNowButton.click();
  }

  const response = await page.request.post("/api/proxy/api/chat/sessions", {
    data: null,
  });
  expect(response.ok()).toBeTruthy();

  const session = await response.json();
  const sessionId = session?.id;
  expect(sessionId).toBeTruthy();

  await page.goto(`/copilot?sessionId=${sessionId}`);
  await expect(page.locator("#chat-input-session")).toBeVisible({
    timeout: 15000,
  });
  await expect(page.getByRole("button", { name: "New Chat" })).toBeVisible({
    timeout: 15000,
  });

  await page.reload();
  await page.waitForLoadState("domcontentloaded");

  await expect
    .poll(() => new URL(page.url()).searchParams.get("sessionId"), {
      timeout: 15000,
    })
    .toBe(sessionId);
  await expect(page.locator("#chat-input-session")).toBeVisible({
    timeout: 15000,
  });
});
