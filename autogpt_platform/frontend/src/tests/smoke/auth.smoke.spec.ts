import { expect, test } from "../coverage-fixture";
import { getSeededTestUser } from "../credentials/accounts";
import { COOKIE_CONSENT_STORAGE_STATE } from "../credentials/storage-state";
import { BuildPage } from "../pages/build.page";
import { LoginPage } from "../pages/login.page";
import { completeOnboardingWizard } from "../utils/onboarding";
import { signupTestUser } from "../utils/signup";

test.use({ storageState: COOKIE_CONSENT_STORAGE_STATE });

test("@smoke auth flow: user can sign up and complete onboarding", async ({
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

test("@smoke auth flow: seeded user can log in and log out across multiple tabs", async ({
  context,
}) => {
  test.setTimeout(90000);

  const testUser = getSeededTestUser("smokeAuth");
  const consoleErrors: string[] = [];

  const page1 = await context.newPage();
  const page2 = await context.newPage();
  const loginPage = new LoginPage(page1);
  const buildPage = new BuildPage(page1);

  for (const page of [page1, page2]) {
    page.on("console", (message) => {
      if (message.type() === "error" && message.text().includes("WebSocket")) {
        consoleErrors.push(message.text());
      }
    });
  }

  await page1.goto("/login");
  await loginPage.login(testUser.email, testUser.password);
  await expect(page1).toHaveURL(/\/marketplace/);

  await page1.goto("/build");
  await buildPage.closeTutorial();
  await expect(page1.getByTestId("profile-popout-menu-trigger")).toBeVisible();

  await page2.goto("/build");
  await expect(page2.getByTestId("profile-popout-menu-trigger")).toBeVisible();

  await page1.getByTestId("profile-popout-menu-trigger").click();
  await page1.getByRole("button", { name: "Log out" }).click();
  await expect(page1).toHaveURL(/\/login/);

  await page2.waitForTimeout(2000);
  await page2.reload();
  await expect(page2).toHaveURL(/\/login\?next=%2Fbuild/);
  await expect(page2.getByTestId("profile-popout-menu-trigger")).toHaveCount(0);
  expect(consoleErrors).toHaveLength(0);
});
