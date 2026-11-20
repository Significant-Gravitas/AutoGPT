import { expect, test } from "./coverage-fixture";
import { getSeededTestUser } from "./credentials/accounts";

test("microsoft sign-up: onboarding opens on the Microsoft 365 offer", async ({
  page,
  baseURL,
}) => {
  test.setTimeout(90000);
  const user = getSeededTestUser("microsoftSignIn");

  // Signs in through the API, since this test is about onboarding: the login
  // form can be replaced just after it loads, dropping a typed email. Not
  // LoginPage.login either, which completes onboarding on the way past it.
  const signIn = await page.request.post("/api/auth/sign-in/email", {
    headers: { Origin: new URL(baseURL ?? "http://localhost:3000").origin },
    data: { email: user.email, password: user.password },
  });
  expect(signIn.ok(), `sign-in failed: ${signIn.status()}`).toBe(true);

  // Start onboarding over, so a retry doesn't resume or skip the last attempt.
  const reset = await page.request.post("/api/proxy/api/onboarding/reset");
  expect(reset.ok()).toBe(true);
  await page.goto("/onboarding");

  await expect(
    page.getByRole("heading", { name: "Also connect Microsoft 365?" }),
  ).toBeVisible({ timeout: 15000 });
  await page.getByRole("button", { name: "Not now" }).click();
  await expect(page).toHaveURL(/\/onboarding\?step=team/);
});
