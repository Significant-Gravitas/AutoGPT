import { Locator, Page } from "@playwright/test";

export class AgentNotificationsPage {
  constructor(private page: Page) {}

  get notificationButton(): Locator {
    return this.page.locator('button[title="Agent Activity"]');
  }

  get notificationBadge(): Locator {
    return this.page
      .locator('button[title="Agent Activity"] .animate-spin')
      .first();
  }

  get notificationDropdown(): Locator {
    return this.page.locator('[role="dialog"]:has-text("Agent Activity")');
  }

  get notificationItems(): Locator {
    return this.notificationDropdown.locator('[role="button"]');
  }

  async clickNotificationButton(): Promise<void> {
    await this.notificationButton.click();
  }

  async isNotificationBadgeVisible(): Promise<boolean> {
    return await this.notificationBadge.isVisible();
  }

  async isNotificationDropdownVisible(): Promise<boolean> {
    return await this.notificationDropdown.isVisible();
  }

  async getNotificationCount(): Promise<string> {
    const badge = this.page.locator(
      'button[title="Agent Activity"] .bg-purple-600',
    );
    return (await badge.textContent()) || "0";
  }

  async getNotificationItems(): Promise<
    { name: string; status: string; time: string }[]
  > {
    const items = await this.notificationItems.all();
    const results = [];

    for (const item of items) {
      const name = (await item.locator(".truncate").textContent()) || "";
      const time =
        (await item.locator(".\\!text-zinc-500").textContent()) || "";

      // Determine status from icon classes and text content
      let status = "unknown";
      if (await item.locator(".animate-spin").isVisible()) {
        status = "running";
      } else if (await item.locator("svg").first().isVisible()) {
        // For non-animated icons, check the text content to determine status
        const timeText = time.toLowerCase();
        if (timeText.includes("completed")) {
          status = "completed";
        } else if (timeText.includes("failed")) {
          status = "failed";
        } else if (timeText.includes("stopped")) {
          status = "terminated";
        } else if (timeText.includes("incomplete")) {
          status = "incomplete";
        } else if (timeText.includes("queued")) {
          status = "queued";
        }
      }

      results.push({ name, status, time });
    }

    return results;
  }

  async waitForNotificationUpdate(_timeout = 10000): Promise<void> {
    await this.page.waitForTimeout(1000); // Wait for potential updates
  }

  async hasNotificationWithStatus(status: string): Promise<boolean> {
    const items = await this.getNotificationItems();
    return items.some((item) => item.status === status);
  }

  async getNotificationByAgentName(
    agentName: string,
  ): Promise<{ name: string; status: string; time: string } | null> {
    const items = await this.getNotificationItems();
    return items.find((item) => item.name.includes(agentName)) || null;
  }
}
