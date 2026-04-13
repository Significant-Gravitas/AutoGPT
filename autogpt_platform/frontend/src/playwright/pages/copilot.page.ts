import { expect, Locator, Page } from "@playwright/test";
import { BasePage } from "./base.page";

export class CopilotPage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  async open(sessionId?: string): Promise<void> {
    const url = sessionId ? `/copilot?sessionId=${sessionId}` : "/copilot";
    await this.page.goto(url);
    await expect(this.page).toHaveURL(/\/copilot/);
    await this.dismissNotificationPrompt();
  }

  async dismissNotificationPrompt(): Promise<void> {
    // Notification permission prompt is optional — only shown on first visit
    const notNowButton = this.page.getByRole("button", { name: "Not now" });
    if (await notNowButton.isVisible({ timeout: 3000 })) {
      await notNowButton.click();
    }
  }

  async createSessionViaApi(): Promise<string> {
    const response = await this.page.request.post(
      "/api/proxy/api/chat/sessions",
      { data: null },
    );
    expect(response.ok()).toBeTruthy();

    const session = await response.json();
    const sessionId = session?.id;
    expect(sessionId).toBeTruthy();
    return sessionId as string;
  }

  getChatInput(): Locator {
    return this.page.locator("#chat-input-session");
  }

  getEmptyChatInput(): Locator {
    return this.page.locator("#chat-input-empty");
  }

  getEmptyChatSubmitButton(): Locator {
    return this.page
      .locator("form")
      .filter({ has: this.getEmptyChatInput() })
      .getByRole("button", { name: "Submit" });
  }

  async waitForChatInput(): Promise<void> {
    await expect(this.getChatInput()).toBeVisible({ timeout: 15000 });
  }

  async waitForEmptyChatInput(): Promise<void> {
    await expect(this.getEmptyChatInput()).toBeVisible({ timeout: 15000 });
  }

  async submitEmptyChatPrompt(prompt: string): Promise<void> {
    await this.getEmptyChatInput().fill(prompt);
    await expect(this.getEmptyChatSubmitButton()).toBeEnabled({
      timeout: 15000,
    });
    await this.getEmptyChatSubmitButton().click();
  }
}
