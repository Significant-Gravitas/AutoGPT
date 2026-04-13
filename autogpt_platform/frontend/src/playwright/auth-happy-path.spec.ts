import { expect, test } from "./coverage-fixture";
import { getSeededTestUser } from "./credentials/accounts";
import { LoginPage } from "./pages/login.page";
import {
  completeOnboardingWizard,
  skipOnboardingIfPresent,
} from "./utils/onboarding";
import { signupTestUser } from "./utils/signup";

test("auth happy path: seeded user can log in and log out", async ({
  page,
}) => {
  test.setTimeout(60000);

  const testUser = getSeededTestUser("smokeAuth");
  const loginPage = new LoginPage(page);

  await page.goto("/login");
  await loginPage.login(testUser.email, testUser.password);
  await skipOnboardingIfPresent(page, "/marketplace");

  await expect(page).toHaveURL(/\/marketplace/);
  await expect(page.getByTestId("profile-popout-menu-trigger")).toBeVisible();
  await page.getByTestId("profile-popout-menu-trigger").click();
  await page.getByRole("button", { name: "Log out" }).click();

  await expect(page).toHaveURL(/\/login/, { timeout: 15000 });

  await page.goto("/library");
  await expect(page).toHaveURL(/\/login\?next=%2Flibrary/);
});

test("auth happy path: user can sign up, complete onboarding, and land in the app", async ({
  page,
}) => {
  test.setTimeout(90000);

  await signupTestUser(page, undefined, undefined, false);
  await expect(page).toHaveURL(/\/onboarding/);

  await completeOnboardingWizard(page, {
    name: "Smoke User",
    role: "Engineering",
    painPoints: ["Research", "Reports & data"],
  });

  await expect(page).toHaveURL(/\/copilot/);
  await expect(page.getByTestId("profile-popout-menu-trigger")).toBeVisible();
  await expect(page.locator("#chat-input-empty")).toBeVisible();
});
