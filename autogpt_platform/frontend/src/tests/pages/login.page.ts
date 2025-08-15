import { Page } from "@playwright/test";

export class LoginPage {
  constructor(private page: Page) {}

  async goto() {
    await this.page.goto("/login");
  }

  async login(email: string, password: string) {
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
    this.page.on("load", (page) => {
      // eslint-disable-next-line no-console
      console.log(`ℹ️ Now at URL: ${page.url()}`);
    });

    // Start waiting for navigation before clicking
    const leaveLoginPage = this.page
      .waitForURL(
        (url) => /^\/(marketplace|onboarding(\/.*)?)?$/.test(url.pathname),
        { timeout: 10_000 },
      )
      .catch((reason) => {
        // eslint-disable-next-line no-console
        console.error(
          `🚨 Navigation away from /login timed out (current URL: ${this.page.url()}):`,
          reason,
        );
        throw reason;
      });

    await loginButton.click();

    await leaveLoginPage;

    await new Promise((resolve) => setTimeout(resolve, 200)); // allow time for client-side redirect
    await this.page.waitForLoadState("load", { timeout: 10_000 });

    await this.page.goto("/marketplace", { timeout: 10_000 });
  }
}
