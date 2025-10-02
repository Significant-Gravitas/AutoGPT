import { Page } from "@playwright/test";

export class LoginPage {
  constructor(private page: Page) {}

  async goto() {
    await this.page.goto("/login");
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
    this.page.on("load", (page) => console.log(`ℹ️ Now at URL: ${page.url()}`));

    // Start waiting for navigation before clicking
    const leaveLoginPage = this.page
      .waitForURL(
        (url) => /^\/(marketplace|onboarding(\/.*)?)?$/.test(url.pathname),
        { timeout: 10_000 },
      )
      .catch((reason) => {
        console.error(
          `🚨 Navigation away from /login timed out (current URL: ${this.page.url()}):`,
          reason,
        );
        throw reason;
      });

    console.log(`🖱️ Clicking login button...`);
    await loginButton.click();

    console.log("⏳ Waiting for navigation away from /login ...");
    await leaveLoginPage;
    console.log(`⌛ Post-login redirected to ${this.page.url()}`);

    await new Promise((resolve) => setTimeout(resolve, 200)); // allow time for client-side redirect
    await this.page.waitForLoadState("load", { timeout: 10_000 });

    console.log("➡️ Navigating to /marketplace ...");
    await this.page.goto("/marketplace", { timeout: 10_000 });
    console.log("✅ Login process complete");

    // If Wallet popover auto-opens, close it to avoid blocking account menu interactions
    try {
      const walletPanel = this.page.getByText("Your credits").first();
      // Wait briefly for wallet to appear after navigation (it may open asynchronously)
      const appeared = await walletPanel
        .waitFor({ state: "visible", timeout: 2500 })
        .then(() => true)
        .catch(() => false);
      if (appeared) {
        const closeWalletButton = this.page.getByRole("button", {
          name: /Close wallet/i,
        });
        await closeWalletButton.click({ timeout: 3000 }).catch(async () => {
          // Fallbacks: try Escape, then click outside
          await this.page.keyboard.press("Escape").catch(() => {});
        });
        await walletPanel
          .waitFor({ state: "hidden", timeout: 3000 })
          .catch(async () => {
            await this.page.mouse.click(5, 5).catch(() => {});
          });
      }
    } catch (_e) {
      // Non-fatal in tests; continue
      console.log("(info) Wallet popover not present or already closed");
    }
  }
}
