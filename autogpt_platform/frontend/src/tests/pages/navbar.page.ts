import { Page } from "@playwright/test";

export class NavBar {
  constructor(private page: Page) {}

  async clickProfileLink() {
    await this.page.getByTestId("profile-popout-menu-trigger").click();
    await this.page.getByRole("link", { name: "Edit profile" }).click();
  }

  async clickBuildLink() {
    const link = this.page.getByTestId("navbar-link-build");
    await link.waitFor({ state: "visible", timeout: 15000 });
    await link.scrollIntoViewIfNeeded();
    await link.click();
    await this.page.waitForURL(/\/build$/, { timeout: 15000 });
  }

  async clickMarketplaceLink() {
    await this.page.getByTestId("navbar-link-marketplace").click();
  }

  async getUserMenuButton() {
    return this.page.getByTestId("profile-popout-menu-trigger");
  }

  async clickUserMenu() {
    await (await this.getUserMenuButton()).click();
  }

  async logout() {
    await this.clickUserMenu();
    await this.page.getByText("Log out").click();
  }

  async isLoggedIn(): Promise<boolean> {
    try {
      await (
        await this.getUserMenuButton()
      ).waitFor({
        state: "visible",
        timeout: 10_000,
      });
      return true;
    } catch {
      return false;
    }
  }
}
