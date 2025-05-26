import { Page } from "@playwright/test";

export class LoginPage {
  constructor(private page: Page) {}

  async login(email: string, password: string) {
    console.log(`Attempting login on ${this.page.url()} with`, {
      email,
      password,
    }); // Debug log

    // Fill email
    const emailInput = this.page.getByPlaceholder("m@example.com");
    await emailInput.waitFor({ state: "visible" });
    await emailInput.fill(email);

    // Fill password
    const passwordInput = this.page.getByTitle("Password");
    await passwordInput.waitFor({ state: "visible" });
    await passwordInput.fill(password);

    // TODO: This is a workaround to wait for the page to load after filling the email and password
    const emailInput2 = this.page.getByPlaceholder("m@example.com");
    await emailInput2.waitFor({ state: "visible" });
    await emailInput2.fill(email);

    // Fill password
    const passwordInput2 = this.page.getByTitle("Password");
    await passwordInput2.waitFor({ state: "visible" });
    await passwordInput2.fill(password);

    // Wait for the button to be ready
    const loginButton = this.page.getByRole("button", {
      name: "Login",
      exact: true,
    });
    await loginButton.waitFor({ state: "visible" });

    // Start waiting for navigation before clicking
    const navigationPromise = Promise.race([
      this.page
        .waitForURL(/^\/(marketplace|onboarding(\/.*)?)?$/, { timeout: 10_000 })
        .catch((reason) => {
          console.warn(
            `Navigation away from /login timed out: ${reason}. Current URL: ${this.page.url()}`,
          );
          throw reason;
        }), // Wait for home page
    ]);

    console.log(`About to click login button on ${this.page.url()}`); // Debug log
    await loginButton.click();

    console.log("Waiting for navigation away from /login"); // Debug log
    await navigationPromise;

    await this.page.goto("/marketplace");

    console.log("Navigation complete, waiting for network idle"); // Debug log
    await this.page.waitForLoadState("load", { timeout: 10_000 });
    console.log("Login process complete"); // Debug log
  }
}
