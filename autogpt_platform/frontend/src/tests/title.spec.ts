import { test, expect } from "./fixtures";

test("has title", async ({ page }) => {
  await page.goto("/");

  // Expect a title "to contain" a substring.
  await expect(page).toHaveTitle(/NextGen AutoGPT/);
});
