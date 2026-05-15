import { expect, Locator, Page } from "@playwright/test";
import { BasePage } from "./base.page";

export class SettingsPage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  async open(): Promise<void> {
    await this.page.goto("/profile/settings");
    await expect(this.page).toHaveURL(/\/profile\/settings/);
    await expect(
      this.page.getByText("Manage your account settings and preferences."),
    ).toBeVisible();
  }

  getAgentRunNotificationsSwitch(): Locator {
    return this.page.getByRole("switch", {
      name: "Agent Run Notifications",
    });
  }

  async savePreferences(): Promise<void> {
    await this.page.getByRole("button", { name: "Save preferences" }).click();
    await expect(
      this.page.getByText("Successfully updated notification preferences"),
    ).toBeVisible({ timeout: 15000 });
  }
}
