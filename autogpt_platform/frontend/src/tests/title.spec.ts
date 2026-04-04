import { test, expect } from "./coverage-fixture";

test("has title", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveTitle(/AutoGPT Platform/);
});
