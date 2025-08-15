import { Page } from "@playwright/test";
import { LoginPage } from "../pages/login.page";

type TestUser = {
  email: string;
  password: string;
  id?: string;
};

/**
 * Utility functions for signin/authentication tests
 */
export class SigninUtils {
  constructor(
    private page: Page,
    private loginPage: LoginPage,
  ) {}

  /**
   * Perform login and verify success
   */
  async loginAndVerify(testUser: TestUser): Promise<void> {
    await this.page.goto("/login");
    await this.loginPage.login(testUser.email, testUser.password);

    // Verify we're on marketplace
    await this.page.waitForURL("/marketplace");

    // Verify profile menu is visible (user is authenticated)
    await this.page.getByTestId("profile-popout-menu-trigger").waitFor({
      state: "visible",
      timeout: 5000,
    });
  }

  /**
   * Perform logout and verify success
   */
  async logoutAndVerify(): Promise<void> {
    // Open profile menu
    await this.page.getByTestId("profile-popout-menu-trigger").click();

    // Wait for menu to be visible
    await this.page.getByRole("button", { name: "Log out" }).waitFor({
      state: "visible",
      timeout: 5000,
    });

    // Click logout
    await this.page.getByRole("button", { name: "Log out" }).click();

    // Verify we're back on login page
    await this.page.waitForURL("/login");
  }

  /**
   * Complete authentication cycle: login -> logout -> login
   */
  async fullAuthenticationCycle(testUser: TestUser): Promise<void> {
    // First login
    await this.loginAndVerify(testUser);

    // Logout
    await this.logoutAndVerify();

    // Login again
    await this.loginAndVerify(testUser);
  }

  /**
   * Verify user is on marketplace and authenticated
   */
  async verifyAuthenticated(): Promise<void> {
    await this.page.waitForURL("/marketplace");
    await this.page.getByTestId("profile-popout-menu-trigger").waitFor({
      state: "visible",
      timeout: 5000,
    });
  }

  /**
   * Verify user is on login page (not authenticated)
   */
  async verifyNotAuthenticated(): Promise<void> {
    await this.page.waitForURL("/login");
  }
}
