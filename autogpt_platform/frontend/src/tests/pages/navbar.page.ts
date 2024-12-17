import { Page } from "@playwright/test";

export class NavBar {
  constructor(private page: Page) {}

  async clickProfileLink() {
    await this.page.getByTestId("profile-popout-menu-trigger").click();
    await this.page.getByRole("link", { name: "Edit profile" }).click();
  }

  async clickMonitorLink() {
    await this.page.getByRole("link", { name: "Library" }).click();
  }

  async clickBuildLink() {
    await this.page.locator('a[href="/build"] div').click();
  }

  async clickMarketplaceLink() {
    await this.page.locator('a[href="/store"]').click();
  }

  async getUserMenuButton() {
    return this.page.getByTestId("profile-popout-menu-trigger");
  }

  async clickUserMenu() {
    await this.page.getByTestId("profile-popout-menu-trigger").click();
  }

  async logout() {
    await this.clickUserMenu();
    await this.page.getByText("Log out").click();
  }

  async isLoggedIn(): Promise<boolean> {
    try {
      await this.page.getByTestId("profile-popout-menu-trigger").waitFor({
        state: "visible",
        timeout: 10_000,
      });
      return true;
    } catch {
      return false;
    }
  }
}
