import { Page } from "@playwright/test";
import { NavBar } from "./navbar.page";

export class BasePage {
  readonly navbar: NavBar;

  constructor(protected page: Page) {
    this.navbar = new NavBar(page);
  }

  async waitForPageLoad() {
    // Common page load waiting logic
    await this.page.waitForLoadState("networkidle", { timeout: 10000 });
  }
}
