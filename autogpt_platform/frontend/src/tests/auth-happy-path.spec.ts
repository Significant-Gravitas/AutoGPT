import { expect, test } from "./coverage-fixture";
import { getSeededTestUser } from "./credentials/accounts";
import { COOKIE_CONSENT_STORAGE_STATE } from "./credentials/storage-state";
import { LoginPage } from "./pages/login.page";
import { completeOnboardingWizard } from "./utils/onboarding";
import { signupTestUser } from "./utils/signup";

test.use({ storageState: COOKIE_CONSENT_STORAGE_STATE });

test("auth happy path: user can sign up with a fresh account", async ({
  page,
}) => {
  test.setTimeout(90000);

  await signupTestUser(page, undefined, undefined, false);
  await expect(page).toHaveURL(/\/onboarding/);
  await expect(page.getByText("Welcome to AutoGPT")).toBeVisible();
});

test("auth happy path: seeded user can log in", async ({ page }) => {
  test.setTimeout(90000);

  const testUser = getSeededTestUser("smokeAuth");
  const loginPage = new LoginPage(page);

  await page.goto("/login");
  await loginPage.login(testUser.email, testUser.password);

  await expect(page).toHaveURL(/\/marketplace/);
  await expect(page.getByTestId("profile-popout-menu-trigger")).toBeVisible();
});

test("auth happy path: user can complete onboarding and land in the app", async ({
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
});
