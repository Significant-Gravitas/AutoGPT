import { Page } from "@playwright/test";
import { BasePage } from "./base.page";

// --8<-- [start:ProfilePageExample]
export class ProfilePage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  async getDisplayedHandle(): Promise<string> {
    await this.waitForPageToLoad();
    const handle = await this.page.locator('input[name="handle"]').inputValue();
    if (!handle) {
      throw new Error("Handle not found");
    }
    return handle;
  }

  async getDisplayedName(): Promise<string> {
    await this.waitForPageToLoad();
    const displayName = await this.page
      .locator('input[name="displayName"]')
      .inputValue();
    if (!displayName) {
      throw new Error("Display name not found");
    }
    return displayName;
  }
  // --8<-- [end:ProfilePageExample]
  async isLoaded(): Promise<boolean> {
    try {
      await this.waitForPageToLoad();
      return true;
    } catch (error) {
      console.error("Error loading profile page", error);
      return false;
    }
  }

  private async waitForPageToLoad(): Promise<void> {
    await this.page.waitForLoadState("domcontentloaded", { timeout: 60_000 });

    await this.page.locator('input[name="handle"]').waitFor({
      state: "visible",
      timeout: 10_000,
    });

    await this.page.waitForLoadState("domcontentloaded", { timeout: 60_000 });
  }
}
