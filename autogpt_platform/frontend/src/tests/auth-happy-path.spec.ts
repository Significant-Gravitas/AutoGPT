import { expect, test } from "./coverage-fixture";
import { getSeededTestUser } from "./credentials/accounts";
import { COOKIE_CONSENT_STORAGE_STATE } from "./credentials/storage-state";
import { BuildPage } from "./pages/build.page";
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

test("auth happy path: seeded user can log out and protected routes redirect to login", async ({
  page,
}) => {
  test.setTimeout(90000);

  const testUser = getSeededTestUser("smokeAuth");
  const loginPage = new LoginPage(page);

  await page.goto("/login");
  await loginPage.login(testUser.email, testUser.password);

  await expect(page).toHaveURL(/\/marketplace/);
  await page.getByTestId("profile-popout-menu-trigger").click();
  await page.getByRole("button", { name: "Log out" }).click();

  await expect(page).toHaveURL(/\/login/);

  await page.goto("/profile");
  await expect(page).toHaveURL(/\/login\?next=%2Fprofile/);
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

test("auth happy path: multi-tab logout clears shared builder sessions", async ({
  context,
}) => {
  test.setTimeout(120000);

  const testUser = getSeededTestUser("smokeBuilder");
  const consoleErrors: string[] = [];

  const page1 = await context.newPage();
  const page2 = await context.newPage();
  const loginPage = new LoginPage(page1);
  const buildPage = new BuildPage(page1);

  const recordWebSocketErrors =
    (label: string) => (msg: { type: () => string; text: () => string }) => {
      if (msg.type() === "error" && msg.text().includes("WebSocket")) {
        consoleErrors.push(`${label}: ${msg.text()}`);
      }
    };

  page1.on("console", recordWebSocketErrors("page1"));
  page2.on("console", recordWebSocketErrors("page2"));

  await page1.goto("/login");
  await loginPage.login(testUser.email, testUser.password);
  await expect(page1).toHaveURL(/\/marketplace/);

  await page1.goto("/build");
  await expect(page1).toHaveURL(/\/build/);
  await buildPage.closeTutorial();
  await expect(page1.getByTestId("profile-popout-menu-trigger")).toBeVisible();

  await page2.goto("/build");
  await expect(page2).toHaveURL(/\/build/);
  await expect(page2.getByTestId("profile-popout-menu-trigger")).toBeVisible();

  await page1.getByTestId("profile-popout-menu-trigger").click();
  await page1.getByRole("button", { name: "Log out" }).click();
  await expect(page1).toHaveURL(/\/login/);

  await page2.reload();
  await expect(page2).toHaveURL(/\/login\?next=%2Fbuild/);
  await expect(page2.getByTestId("profile-popout-menu-trigger")).toBeHidden();

  expect(consoleErrors).toHaveLength(0);

  // Prove the auth token is actually gone, not just the UI hidden. Supabase
  // does not delete the cookie on signout — it overwrites it with an empty
  // value and a past expiry. Either form is "cleared"; a non-empty value
  // would mean a regression where the session token still resides in the
  // shared browser context after logout.
  const cookiesAfterLogout = await context.cookies();
  const authCookie = cookiesAfterLogout.find(
    (c) => c.name.startsWith("sb-") && c.name.includes("auth-token"),
  );
  if (authCookie !== undefined) {
    expect(
      authCookie.value,
      "supabase auth cookie must be empty after logout",
    ).toBe("");
  }

  await page1.close();
  await page2.close();
});
