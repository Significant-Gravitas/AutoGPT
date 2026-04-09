import { Page } from "@playwright/test";
import {
  getSeededTestUser,
  type SeededTestAccountKey,
} from "../credentials/accounts";
import { skipOnboardingIfPresent } from "../utils/onboarding";

export class LoginPage {
  constructor(private page: Page) {}

  async goto() {
    await this.page.goto("/login");
  }

  async loginAsSeededUser(userKey: SeededTestAccountKey): Promise<void> {
    const user = getSeededTestUser(userKey);
    await this.page.goto("/login");
    await this.login(user.email, user.password);
  }

  async login(email: string, password: string) {
    console.log(`ℹ️ Attempting login on ${this.page.url()} with`, {
      email,
      password,
    });

    // Wait for the form to be ready
    await this.page.waitForSelector("form", { state: "visible" });

    // Fill email using input selector instead of label
    const emailInput = this.page.locator('input[type="email"]');
    await emailInput.waitFor({ state: "visible" });
    await emailInput.fill(email);

    // Fill password using input selector instead of label
    const passwordInput = this.page.locator('input[type="password"]');
    await passwordInput.waitFor({ state: "visible" });
    await passwordInput.fill(password);

    // Wait for the button to be ready
    const loginButton = this.page.getByRole("button", {
      name: "Login",
      exact: true,
    });
    await loginButton.waitFor({ state: "visible" });

    // Attach navigation logger for debug purposes
    this.page.once("load", (page) =>
      console.log(`ℹ️ Now at URL: ${page.url()}`),
    );

    const hasReachedPostLoginRoute = () =>
      this.page.waitForFunction(
        () => {
          const pathname = window.location.pathname;
          return /^\/(marketplace|onboarding(\/.*)?|library|copilot)$/.test(
            pathname,
          );
        },
        { timeout: 15_000 },
      );

    console.log(`🖱️ Clicking login button...`);
    for (let attempt = 0; attempt < 2; attempt += 1) {
      await loginButton.click();

      console.log("⏳ Waiting for navigation away from /login ...");
      try {
        await hasReachedPostLoginRoute();
        break;
      } catch (reason) {
        const currentPathname = new URL(this.page.url()).pathname;
        if (attempt === 1 || currentPathname !== "/login") {
          console.error(
            `🚨 Navigation away from /login timed out (current URL: ${this.page.url()}):`,
            reason,
          );
          throw reason;
        }
      }
    }

    console.log(`⌛ Post-login redirected to ${this.page.url()}`);

    await this.page.waitForLoadState("load", { timeout: 10_000 });

    // If redirected to onboarding, complete it via API so tests can proceed
    await skipOnboardingIfPresent(this.page, "/marketplace");

    console.log("➡️ Navigating to /marketplace ...");
    await this.page.goto("/marketplace", { timeout: 20_000 });
    console.log("✅ Login process complete");

    // If Wallet popover auto-opens, close it to avoid blocking account menu interactions.
    // The popover is genuinely optional — only appears on some accounts/environments.
    const walletPanel = this.page.getByText("Your credits").first();
    const walletPanelVisible = await walletPanel
      .waitFor({ state: "visible", timeout: 2500 })
      .then(() => true)
      .catch(() => false);
    if (walletPanelVisible) {
      const closeWalletButton = this.page.getByRole("button", {
        name: /Close wallet/i,
      });
      const closeWalletButtonVisible = await closeWalletButton
        .waitFor({ state: "visible", timeout: 1000 })
        .then(() => true)
        .catch(() => false);
      if (closeWalletButtonVisible) {
        await closeWalletButton.click();
      } else {
        await this.page.keyboard.press("Escape");
      }
      const walletStillVisible = await walletPanel
        .waitFor({ state: "hidden", timeout: 3000 })
        .then(() => false)
        .catch(() => true);
      if (walletStillVisible) {
        await this.page.mouse.click(5, 5);
      }
    }
  }
}
