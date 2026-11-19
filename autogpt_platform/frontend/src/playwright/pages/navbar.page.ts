import { Page } from "@playwright/test";

// Global navigation lives in the app sidebar; the account menu trigger sits in
// its footer.
export class NavBar {
  constructor(private page: Page) {}

  private sidebarLink(name: string) {
    return this.page
      .locator('[data-sidebar="sidebar"]')
      .getByRole("link", { name, exact: true });
  }

  async clickProfileLink() {
    await this.page.getByTestId("profile-popout-menu-trigger").click();
    await this.page.getByRole("link", { name: "Profile" }).click();
  }

  async clickBuildLink() {
    const link = this.sidebarLink("Build");
    await link.waitFor({ state: "visible", timeout: 15000 });
    await link.scrollIntoViewIfNeeded();
    await link.click();
    await this.page.waitForURL(/\/build$/, { timeout: 15000 });
  }

  async clickMarketplaceLink() {
    await this.sidebarLink("Marketplace").click();
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
