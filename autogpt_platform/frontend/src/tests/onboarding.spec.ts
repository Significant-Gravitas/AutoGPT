import test, { expect } from "@playwright/test";
import { signupTestUser } from "./utils/signup";
import { completeOnboardingWizard } from "./utils/onboarding";
import { getSelectors } from "./utils/selectors";

test("new user completes full onboarding wizard", async ({ page }) => {
  // Signup WITHOUT skipping onboarding (ignoreOnboarding=false)
  await signupTestUser(page, undefined, undefined, false);

  // Should be on onboarding
  await expect(page).toHaveURL(/\/onboarding/);

  // Complete the wizard
  await completeOnboardingWizard(page, {
    name: "Alice",
    role: "Marketing",
    painPoints: ["Social media", "Email & outreach"],
  });

  // Should have been redirected to /copilot
  await expect(page).toHaveURL(/\/copilot/);

  // User should be authenticated
  await page
    .getByTestId("profile-popout-menu-trigger")
    .waitFor({ state: "visible", timeout: 10000 });
});

test("onboarding wizard step navigation works", async ({ page }) => {
  await signupTestUser(page, undefined, undefined, false);
  await expect(page).toHaveURL(/\/onboarding/);

  // Step 1: Welcome
  await expect(page.getByText("Welcome to AutoGPT")).toBeVisible();
  await page.getByLabel("Your first name").fill("Bob");
  await page.getByRole("button", { name: "Continue" }).click();

  // Step 2: Role — verify we're here, then go back
  await expect(page.getByText("What best describes you")).toBeVisible();
  await page.getByText("Back").click();

  // Should be back on step 1 with name preserved
  await expect(page.getByText("Welcome to AutoGPT")).toBeVisible();
  await expect(page.getByLabel("Your first name")).toHaveValue("Bob");
});

test("onboarding wizard validates required fields", async ({ page }) => {
  await signupTestUser(page, undefined, undefined, false);
  await expect(page).toHaveURL(/\/onboarding/);

  // Step 1: Continue should be disabled without a name
  const continueButton = page.getByRole("button", { name: "Continue" });
  await expect(continueButton).toBeDisabled();

  // Fill name — continue should become enabled
  await page.getByLabel("Your first name").fill("Charlie");
  await expect(continueButton).toBeEnabled();
  await continueButton.click();

  // Step 2: Continue should be disabled without a role
  const step2Continue = page.getByRole("button", { name: "Continue" });
  await expect(step2Continue).toBeDisabled();

  // Select role — continue should become enabled
  await page.getByText("Engineering").click();
  await expect(step2Continue).toBeEnabled();
  await step2Continue.click();

  // Step 3: Launch Autopilot should be disabled without any pain points
  const launchButton = page.getByRole("button", { name: "Launch Autopilot" });
  await expect(launchButton).toBeDisabled();

  // Select a pain point — button should become enabled
  await page.getByText("Research", { exact: true }).click();
  await expect(launchButton).toBeEnabled();
});

test("completed onboarding redirects away from /onboarding", async ({
  page,
}) => {
  // Create user and complete onboarding
  await signupTestUser(page, undefined, undefined, false);
  await completeOnboardingWizard(page);

  // Try to navigate back to onboarding — should be redirected to /copilot
  await page.goto("http://localhost:3000/onboarding");
  await page.waitForURL(/\/copilot/, { timeout: 10000 });
});

test("onboarding URL params sync with steps", async ({ page }) => {
  await signupTestUser(page, undefined, undefined, false);
  await expect(page).toHaveURL(/\/onboarding/);

  // Step 1: URL may or may not include step=1 on initial load (no param is equivalent to step 1)
  await expect(page.getByText("Welcome to AutoGPT")).toBeVisible();

  // Fill name and go to step 2
  await page.getByLabel("Your first name").fill("Test");
  await page.getByRole("button", { name: "Continue" }).click();

  // URL should show step=2
  await expect(page).toHaveURL(/step=2/);
});

test("role-based pain point ordering works", async ({ page }) => {
  await signupTestUser(page, undefined, undefined, false);

  // Complete step 1
  await page.getByLabel("Your first name").fill("Test");
  await page.getByRole("button", { name: "Continue" }).click();

  // Select Sales/BD role
  await page.getByText("Sales / BD").click();
  await page.getByRole("button", { name: "Continue" }).click();

  // On pain points step, "Finding leads" should be visible (top pick for Sales)
  await expect(page.getByText("What's eating your time?")).toBeVisible();
  const { getText } = getSelectors(page);
  await expect(getText("Finding leads")).toBeVisible();
});
