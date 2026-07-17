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
    // The dialog mounts after React hydrates + reads localStorage; on slow CI
    // it can appear AFTER the call returned, leaving a fixed overlay that
    // blocks the chat input. Wait for it to either mount or stay absent.
    const dialog = this.page.getByRole("dialog").filter({
      has: this.page.getByText("Stay in the loop"),
    });
    try {
      await dialog.waitFor({ state: "visible", timeout: 3000 });
    } catch {
      return;
    }
    await this.page
      .getByRole("button", { name: "Not now" })
      .click({ timeout: 3000 });
    await dialog.waitFor({ state: "hidden", timeout: 5000 });
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

  async waitForChatInput(): Promise<void> {
    await expect(this.getChatInput()).toBeVisible({ timeout: 15000 });
  }
}
