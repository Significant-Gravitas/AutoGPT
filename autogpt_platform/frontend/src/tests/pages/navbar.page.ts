import { Page } from "@playwright/test";

export class NavBar {
  constructor(private page: Page) {}

  async clickProfileLink() {
    // await this.page.getByTestId("profile-link").click();

    await this.page.getByRole("button", { name: "CN" }).click();
    await this.page.getByRole("menuitem", { name: "Profile" }).click();
  }

  async clickMonitorLink() {
    await this.page.getByTestId("monitor-link").click();
  }

  async clickBuildLink() {
    await this.page.getByTestId("build-link").click();
  }

  async clickMarketplaceLink() {
    await this.page.getByTestId("marketplace-link").click();
  }

  async getUserMenuButton() {
    return this.page.getByRole("button", { name: "CN" });
  }

  async clickUserMenu() {
    await (await this.getUserMenuButton()).click();
  }

  async logout() {
    await this.clickUserMenu();
    await this.page.getByRole("menuitem", { name: "Log out" }).click();
  }

  async isLoggedIn(): Promise<boolean> {
    try {
      await (
        await this.getUserMenuButton()
      ).waitFor({
        state: "visible",
        timeout: 5000,
      });
      return true;
    } catch {
      return false;
    }
  }
}
