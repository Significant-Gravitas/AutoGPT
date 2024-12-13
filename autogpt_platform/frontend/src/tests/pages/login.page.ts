import { Page } from "@playwright/test";

export class LoginPage {
  constructor(private page: Page) {}

  async login(email: string, password: string) {
    console.log("Attempting login with:", { email, password }); // Debug log

    // Fill email
    const emailInput = this.page.getByPlaceholder("user@email.com");
    await emailInput.waitFor({ state: "visible" });
    await emailInput.fill(email);

    // Fill password
    const passwordInput = this.page.getByPlaceholder("password");
    await passwordInput.waitFor({ state: "visible" });
    await passwordInput.fill(password);

    // Check terms
    const termsCheckbox = this.page.getByLabel("I agree to the Terms of Use");
    await termsCheckbox.waitFor({ state: "visible" });
    await termsCheckbox.click();

    // TODO: This is a workaround to wait for the page to load after filling the email and password
    const emailInput2 = this.page.getByPlaceholder("user@email.com");
    await emailInput2.waitFor({ state: "visible" });
    await emailInput2.fill(email);

    // Fill password
    const passwordInput2 = this.page.getByPlaceholder("password");
    await passwordInput2.waitFor({ state: "visible" });
    await passwordInput2.fill(password);

    // Wait for the button to be ready
    const loginButton = this.page.getByRole("button", {
      name: "Log in",
      exact: true,
    });
    await loginButton.waitFor({ state: "visible" });

    // Start waiting for navigation before clicking
    const navigationPromise = this.page.waitForURL("/", { timeout: 10_000 });

    console.log("About to click login button"); // Debug log
    await loginButton.click();

    console.log("Waiting for navigation"); // Debug log
    await navigationPromise;

    console.log("Navigation complete, waiting for network idle"); // Debug log
    await this.page.waitForLoadState("load", { timeout: 10_000 });
    console.log("Login process complete"); // Debug log
  }
}
