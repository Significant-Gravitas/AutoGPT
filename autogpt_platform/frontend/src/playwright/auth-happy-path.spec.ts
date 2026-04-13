import { expect, test } from "./coverage-fixture";
import { getSeededTestUser } from "./credentials/accounts";
import { BuildPage } from "./pages/build.page";
import { LoginPage } from "./pages/login.page";
import {
  completeOnboardingWizard,
  skipOnboardingIfPresent,
} from "./utils/onboarding";
import { signupTestUser } from "./utils/signup";

test("auth happy path: user can sign up with a fresh account", async ({
  page,
}) => {
  test.setTimeout(60000);

  await signupTestUser(page, undefined, undefined, false);
  await expect(page).toHaveURL(/\/onboarding/);
  await expect(page.getByText("Welcome to AutoGPT")).toBeVisible();
});

test("auth happy path: user can sign up, enter the app, and log out", async ({
  page,
}) => {
  test.setTimeout(90000);

  await signupTestUser(page, undefined, undefined, false);
  await expect(page).toHaveURL(/\/onboarding/);
  await expect(page.getByText("Welcome to AutoGPT")).toBeVisible();

  await skipOnboardingIfPresent(page, "/marketplace");
  await expect(page).toHaveURL(/\/marketplace/);
  await expect(page.getByTestId("profile-popout-menu-trigger")).toBeVisible();

  await page.getByTestId("profile-popout-menu-trigger").click();
  await page.getByRole("button", { name: "Log out" }).click();

  await expect(page).toHaveURL(/\/login/);

  await page.goto("/library");
  await expect(page).toHaveURL(/\/login\?next=%2Flibrary/);
});

test("auth happy path: seeded user can log in", async ({ page }) => {
  test.setTimeout(60000);

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
  test.setTimeout(60000);

  const testUser = getSeededTestUser("primary");
  const loginPage = new LoginPage(page);

  await page.goto("/login");
  await loginPage.login(testUser.email, testUser.password);

  await expect(page).toHaveURL(/\/marketplace/);
  await page.getByTestId("profile-popout-menu-trigger").click();
  await page.getByRole("button", { name: "Log out" }).click();

  await expect(page).toHaveURL(/\/login/, { timeout: 15000 });

  await page.goto("/profile");
  await expect(page).toHaveURL(/\/login\?next=%2Fprofile/);
});

test("auth happy path: user can complete onboarding and land in the app", async ({
  page,
}) => {
  test.setTimeout(60000);

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
  // Two pages + builder load + logout sequence justifies a higher timeout
  test.setTimeout(90000);

  const consoleErrors: string[] = [];

  const page1 = await context.newPage();
  const page2 = await context.newPage();
  const buildPage = new BuildPage(page1);

  const recordWebSocketErrors =
    (label: string) => (msg: { type: () => string; text: () => string }) => {
      if (msg.type() === "error" && msg.text().includes("WebSocket")) {
        consoleErrors.push(`${label}: ${msg.text()}`);
      }
    };

  page1.on("console", recordWebSocketErrors("page1"));
  page2.on("console", recordWebSocketErrors("page2"));

  await signupTestUser(page1, undefined, undefined, false);
  await expect(page1).toHaveURL(/\/onboarding/);
  await skipOnboardingIfPresent(page1, "/build");

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
  // overwrites the cookie on signout with an empty value + past expiry
  // rather than deleting it. An assertion that is silently skipped when the
  // cookie is missing under the expected name would hide a real regression,
  // so we assert on every non-empty sb-*auth-token* cookie explicitly.
  const cookiesAfterLogout = await context.cookies();
  const authCookies = cookiesAfterLogout.filter(
    (c) => c.name.startsWith("sb-") && c.name.includes("auth-token"),
  );
  for (const cookie of authCookies) {
    expect(
      cookie.value,
      `supabase auth cookie ${cookie.name} must be empty after logout`,
    ).toBe("");
  }

  await page1.close();
  await page2.close();
});
