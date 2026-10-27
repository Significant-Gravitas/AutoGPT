import { expect, test } from "./coverage-fixture";
import { getSeededTestUser } from "./credentials/accounts";

test("microsoft sign-up: onboarding opens on the Microsoft 365 offer", async ({
  page,
}) => {
  test.setTimeout(90000);
  const user = getSeededTestUser("microsoftSignIn");

  // Not LoginPage.login, which completes onboarding on the way past it.
  await page.goto("/login");
  await page.locator('input[type="email"]').fill(user.email);
  await page.locator('input[type="password"]').fill(user.password);
  await page.getByRole("button", { name: "Log in", exact: true }).click();
  await page.waitForURL((url) => url.pathname !== "/login");

  // Start onboarding over, so a retry doesn't resume or skip the last attempt.
  const reset = await page.request.post(
    new URL("/api/proxy/api/onboarding/reset", page.url()).toString(),
  );
  expect(reset.ok()).toBe(true);
  await page.goto("/onboarding");

  await expect(
    page.getByRole("heading", { name: "Also connect Microsoft 365?" }),
  ).toBeVisible({ timeout: 15000 });
  await page.getByRole("button", { name: "Not now" }).click();
  await expect(page).toHaveURL(/\/onboarding\?step=team/);
});
