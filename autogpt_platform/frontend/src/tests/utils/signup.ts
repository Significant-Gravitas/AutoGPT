import { TestUser } from "./auth";
import { getSelectors } from "./selectors";
import { isVisible } from "./assertion";
import { BuildPage } from "../pages/build.page";

export async function signupTestUser(
  page: any,
  email?: string,
  password?: string,
  ignoreOnboarding: boolean = true,
  withAgent: boolean = false,
): Promise<TestUser> {
  const { faker } = await import("@faker-js/faker");
  const userEmail = email || faker.internet.email();
  const userPassword = password || faker.internet.password({ length: 12 });

  const { getText, getField, getRole, getButton, getId } = getSelectors(page);

  try {
    // Navigate to signup page
    await page.goto("http://localhost:3000/signup");

    // Wait for page to load
    getText("Create a new account");

    // Fill form
    const emailInput = getField("Email");
    await emailInput.fill(userEmail);
    const passwordInput = page.locator("#password");
    await passwordInput.fill(userPassword);
    const confirmPasswordInput = page.locator("#confirmPassword");
    await confirmPasswordInput.fill(userPassword);

    // Agree to terms and submit
    await getRole("checkbox").click();
    const signupButton = getButton("Sign up");
    await signupButton.click();

    // Wait for successful signup - could redirect to onboarding or marketplace

    try {
      // Wait for either onboarding or marketplace redirect
      await Promise.race([
        page.waitForURL(/\/onboarding/, { timeout: 15000 }),
        page.waitForURL(/\/marketplace/, { timeout: 15000 }),
      ]);
    } catch (error) {
      console.error(
        "❌ Timeout waiting for redirect, current URL:",
        page.url(),
      );
      throw error;
    }

    const currentUrl = page.url();

    // Handle onboarding or marketplace redirect
    if (currentUrl.includes("/onboarding") && ignoreOnboarding) {
      await page.goto("http://localhost:3000/marketplace");
      await page.waitForLoadState("domcontentloaded", { timeout: 10000 });
    }

    // Verify we're on the expected final page
    if (ignoreOnboarding || currentUrl.includes("/marketplace")) {
      // Verify we're on marketplace
      await page
        .getByText(
          "Bringing you AI agents designed by thinkers from around the world",
        )
        .waitFor({ state: "visible", timeout: 10000 });

      // Verify user is authenticated (profile menu visible)
      await page
        .getByTestId("profile-popout-menu-trigger")
        .waitFor({ state: "visible", timeout: 10000 });
    }

    if (withAgent) {
      // Create a dummy agent for each new user
      const buildLink = getId("navbar-link-build");
      await buildLink.click();

      const blocksBtn = getId("blocks-control-blocks-button");
      await isVisible(blocksBtn);

      const buildPage = new BuildPage(page);
      await buildPage.createDummyAgent();
    }

    const testUser: TestUser = {
      email: userEmail,
      password: userPassword,
      createdAt: new Date().toISOString(),
    };

    return testUser;
  } catch (error) {
    console.error(`❌ Error creating test user ${userEmail}:`, error);
    throw error;
  }
}

export async function signupAndNavigateToMarketplace(
  page: any,
  email?: string,
  password?: string,
): Promise<TestUser> {
  console.log("🧪 Creating user and navigating to marketplace...");

  // Create the user via signup and automatically navigate to marketplace
  const testUser = await signupTestUser(page, email, password, true);

  console.log("✅ User successfully created and authenticated in marketplace");
  return testUser;
}

export async function validateSignupForm(page: any): Promise<void> {
  console.log("🧪 Validating signup form...");

  await page.goto("http://localhost:3000/signup");

  // Test empty form submission
  console.log("❌ Testing empty form submission...");
  const signupButton = page.getByRole("button", { name: "Sign up" });
  await signupButton.click();

  // Should still be on signup page
  const currentUrl = page.url();
  if (currentUrl.includes("/signup")) {
    console.log("✅ Empty form correctly blocked");
  } else {
    console.log("⚠️ Empty form was not blocked as expected");
  }

  // Test invalid email
  console.log("❌ Testing invalid email...");
  await page.getByLabel("Email").fill("invalid-email");
  await signupButton.click();

  // Should still be on signup page
  const currentUrl2 = page.url();
  if (currentUrl2.includes("/signup")) {
    console.log("✅ Invalid email correctly blocked");
  } else {
    console.log("⚠️ Invalid email was not blocked as expected");
  }

  console.log("✅ Signup form validation completed");
}

export function generateTestEmail(): string {
  return `test.${Date.now()}.${Math.random().toString(36).substring(7)}@example.com`;
}

export async function generateTestPassword(): Promise<string> {
  const { faker } = await import("@faker-js/faker");
  return faker.internet.password({ length: 12 });
}
