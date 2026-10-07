import { TestUser } from "./auth";
import { getSelectors } from "./selectors";
import { isVisible } from "./assertion";
import { BuildPage } from "../pages/build.page";
import { skipOnboardingIfPresent } from "./onboarding";
import { expect } from "@playwright/test";

export function isAuthenticatedAppURL(currentURL: string) {
  const { pathname } = new URL(currentURL, "http://localhost");
  return ["/home", "/copilot", "/library"].some(
    (path) => pathname === path || pathname.startsWith(`${path}/`),
  );
}

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
    await page.goto("/signup");

    // Wait for the hydrated form, not the server-rendered loading shell.
    await getText("Create your account").waitFor({ state: "visible" });

    // Fill form
    const emailInput = getField("Email");
    await emailInput.fill(userEmail);
    await expect(emailInput).toHaveValue(userEmail);
    const passwordInput = page.locator("#password");
    await passwordInput.fill(userPassword);
    const confirmPasswordInput = page.locator("#confirmPassword");
    await confirmPasswordInput.fill(userPassword);

    // Agree to terms and submit. Scope to the Terms checkbox by accessible
    // name — in dev/local the AgentationDevtool renders extra checkboxes
    // globally, so a bare getByRole("checkbox") trips Playwright strict mode.
    await getRole("checkbox", /agree to the terms/i).click();
    const signupButton = getButton("Sign up");
    await signupButton.click();

    // Wait for successful signup - could redirect to various pages depending on onboarding state

    try {
      // Wait for redirect to onboarding, marketplace, copilot, or library
      // Use a single waitForURL with a callback to avoid Promise.race race conditions
      await page.waitForURL(
        (url: URL) =>
          /\/(onboarding|marketplace|copilot|home|library)/.test(url.pathname),
        { timeout: 15000 },
      );
    } catch (error) {
      console.error(
        "❌ Timeout waiting for redirect, current URL:",
        page.url(),
      );
      throw error;
    }

    const currentUrl = page.url();

    // Handle onboarding redirect if needed
    if (currentUrl.includes("/onboarding") && ignoreOnboarding) {
      await skipOnboardingIfPresent(page, "/marketplace");
    }

    // Verify we're on an expected final page and user is authenticated
    if (isAuthenticatedAppURL(currentUrl)) {
      await page
        .getByTestId("profile-popout-menu-trigger")
        .waitFor({ state: "visible", timeout: 10000 });
    } else if (ignoreOnboarding || currentUrl.includes("/marketplace")) {
      await page
        .getByText(
          "Bringing you AI agents designed by thinkers from around the world",
        )
        .first()
        .waitFor({ state: "visible", timeout: 10000 });

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
