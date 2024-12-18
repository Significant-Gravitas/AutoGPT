import { Page } from "@playwright/test";
import { NavBar } from "./navbar.page";

export class BasePage {
  readonly navbar: NavBar;
  readonly downloadsFolder = "./.test-contents";

  constructor(protected page: Page) {
    this.navbar = new NavBar(page);
  }

  async waitForPageLoad() {
    // Common page load waiting logic
    console.log(`waiting for page to load`);
    await this.page.waitForLoadState("domcontentloaded", { timeout: 10_000 });
  }
}
