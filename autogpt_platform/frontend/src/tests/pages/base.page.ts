import { Page } from "@playwright/test";
import { NavBar } from "./navbar.page";

export class BasePage {
  readonly navbar: NavBar;
  readonly downloadsFolder = "./.test-contents";

  constructor(protected page: Page) {
    this.navbar = new NavBar(page);
  }
}
