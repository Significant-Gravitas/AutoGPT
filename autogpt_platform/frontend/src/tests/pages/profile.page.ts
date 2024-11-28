import { Page } from "@playwright/test";
import { BasePage } from "./base.page";

export class ProfilePage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  async getDisplayedEmail(): Promise<string> {
    await this.waitForPageToLoad();
    const email = await this.page.getByTestId("profile-email").textContent();
    if (!email) {
      throw new Error("Email not found");
    }
    return email;
  }

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
    await this.page.waitForLoadState("networkidle", { timeout: 60_000 });

    await this.page.getByTestId("profile-email").waitFor({
      state: "visible",
      timeout: 60_000,
    });

    await this.page.waitForLoadState("networkidle", { timeout: 60_000 });
  }
}
