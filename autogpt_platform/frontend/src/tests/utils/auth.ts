import { faker } from "@faker-js/faker";
import { Page, expect } from "@playwright/test";

export interface TestUser {
  email: string;
  password: string;
  id?: string;
}

/**
 * Create a new test user via the signup process
 */
export async function createTestUser(page: Page): Promise<TestUser> {
  const testUser: TestUser = {
    email: `test.${Date.now()}.${faker.number.int({ min: 1000, max: 9999 })}@example.com`,
    password: faker.internet.password({ length: 12 }),
  };

  console.log(`🚀 Creating test user: ${testUser.email}`);

  // Navigate to signup page
  await page.goto("/signup");

  // Fill out the signup form
  await page.getByPlaceholder("m@example.com").fill(testUser.email);

  // Fill password fields
  const passwordInputs = page.getByTitle("Password");
  await passwordInputs.nth(0).fill(testUser.password); // Password field
  await passwordInputs.nth(1).fill(testUser.password); // Confirm password field

  // Agree to terms
  await page.getByRole("button", { name: "checkbox" }).click();

  // Submit the form
  await page.getByRole("button", { name: "Sign up" }).click();

  // Wait for navigation to onboarding
  await page.waitForURL(/\/onboarding/);

  console.log(`✅ Test user created successfully: ${testUser.email}`);
  return testUser;
}

/**
 * Complete onboarding flow up to step 5, then navigate to marketplace
 */
export async function completeOnboarding(page: Page): Promise<void> {
  console.log("🎯 Starting onboarding completion...");

  // Step 1: Welcome page
  await expect(page.getByText("Welcome to AutoGPT")).toBeVisible();
  await page.getByRole("link", { name: "Continue" }).click();

  // Step 2: Choose reason
  await page.waitForURL("/onboarding/2-reason");
  await expect(
    page.getByText("What's your main reason for using AutoGPT?"),
  ).toBeVisible();

  // Select first reason
  await page.getByText("Content & Marketing").click();
  await page.getByRole("link", { name: "Next" }).click();

  // Step 3: Services/Integrations
  await page.waitForURL("/onboarding/3-services");
  await expect(
    page.getByText(
      "What platforms or services would you like AutoGPT to work with?",
    ),
  ).toBeVisible();

  // Select a couple of integrations
  await page.getByText("Discord").click();
  await page.getByText("GitHub").click();
  await page.getByRole("link", { name: "Next" }).click();

  // Step 4: Choose agent
  await page.waitForURL("/onboarding/4-agent");
  await expect(page.getByText("Choose an agent")).toBeVisible();

  // Wait for agents to load and select the first one
  await page.waitForTimeout(2000); // Wait for agents to load
  const agentCards = page
    .locator('[class*="cursor-pointer"]')
    .filter({ hasText: /runs/ });
  await agentCards.first().click();
  await page.getByRole("link", { name: "Next" }).click();

  // Step 5: Run agent page - this is where we navigate away
  await page.waitForURL("/onboarding/5-run");
  await expect(page.getByText("Run your first agent")).toBeVisible();

  console.log("🛒 Navigating to marketplace at step 5...");
  // Navigate directly to marketplace instead of completing the run
  await page.goto("/marketplace");

  // Verify we're on marketplace and user is ready
  await expect(
    page.getByText(
      "Bringing you AI agents designed by thinkers from around the world",
    ),
  ).toBeVisible();
  await expect(page).toHaveURL("/marketplace");
  await expect(page.getByTestId("profile-popout-menu-trigger")).toBeVisible();

  console.log(
    "✅ Onboarding completed successfully, user ready at marketplace",
  );
}

/**
 * Create and setup a complete test user ready for use
 */
export async function createAndSetupTestUser(page: Page): Promise<TestUser> {
  const testUser = await createTestUser(page);
  await completeOnboarding(page);
  return testUser;
}

export async function loginUser(page: Page, email: string, password: string) {
  await page.goto("/login");

  // Fill email
  const emailInput = page.getByPlaceholder("m@example.com");
  await emailInput.waitFor({ state: "visible" });
  await emailInput.fill(email);

  // Fill password
  const passwordInput = page.getByTitle("Password");
  await passwordInput.waitFor({ state: "visible" });
  await passwordInput.fill(password);

  // Click login button
  const loginButton = page.getByRole("button", { name: "Login", exact: true });
  await loginButton.waitFor({ state: "visible" });
  await loginButton.click();

  // Wait for navigation after login - could be onboarding or marketplace
  await page.waitForURL(/\/(marketplace|onboarding)/);
  const currentUrl = page.url();

  // Check if we landed on onboarding
  if (currentUrl.includes("/onboarding")) {
    // Navigate to marketplace
    console.log("🛒 Navigating to marketplace...");
    await page.goto("/marketplace");
    await page.waitForLoadState("load");

    // Verify we're now on marketplace
    await expect(
      page.getByText(
        "Bringing you AI agents designed by thinkers from around the world",
      ),
    ).toBeVisible();
    await expect(page).toHaveURL("/marketplace");
    // If we landed on marketplace
  } else if (
    currentUrl === "http://localhost:3000/marketplace" ||
    currentUrl.endsWith("/marketplace")
  ) {
    // Verify we're on marketplace
    await expect(
      page.getByText(
        "Bringing you AI agents designed by thinkers from around the world",
      ),
    ).toBeVisible();
    await expect(page).toHaveURL("/marketplace");
  } else {
    throw new Error(`Unexpected landing page after login: ${currentUrl}`);
  }

  // Assert that the profile menu is visible (indicates successful authentication)
  await expect(page.getByTestId("profile-popout-menu-trigger")).toBeVisible();
}

/**
 * Log out the current user
 */
export async function logoutUser(page: Page) {
  console.log("📤 Logging out user...");

  // Click on the profile menu trigger
  await page.getByTestId("profile-popout-menu-trigger").click();

  // Wait for menu to be visible and click logout
  const logoutButton = page.getByRole("button", { name: "Log out" });
  await logoutButton.waitFor({ state: "visible", timeout: 5000 });
  await logoutButton.click();

  // Wait for redirect to login page
  await page.waitForURL("/login");
  console.log("✅ Logout successful");
}
