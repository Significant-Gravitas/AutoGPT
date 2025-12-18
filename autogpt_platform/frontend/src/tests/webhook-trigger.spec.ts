import { expect, test } from "@playwright/test";

const ROUTE = "/playwright/webhook-trigger";

test.describe("Webhook manual block", () => {
  test("wraps long webhook URLs without overflow", async ({ page }) => {
    await page.goto(ROUTE);

    const block = page.locator('[data-blockid="manual-webhook-block"]');
    await expect(block).toBeVisible();

    const webhookUrl = block.locator("code");
    await expect(webhookUrl).toContainText("https://hooks.autogpt.io/");

    await expect(block).toHaveScreenshot("webhook-manual-block.png");
  });
});
