import { Locator, Page } from "@playwright/test";
import { AgentNotificationsPage } from "./agent-notifications.page";

export class LibraryPage {
  public agentNotifications: AgentNotificationsPage;

  constructor(private page: Page) {
    this.agentNotifications = new AgentNotificationsPage(page);
  }

  get libraryTab(): Locator {
    return this.page.locator('a[href="/library"]');
  }

  get agentCards(): Locator {
    return this.page.locator(".agpt-div").filter({ hasText: /^test-agent-/ });
  }

  get runButton(): Locator {
    return this.page.locator('button:has-text("Run")');
  }

  get newRunButton(): Locator {
    return this.page.locator('button:has-text("New run")');
  }

  get runDialogRunButton(): Locator {
    return this.page.locator('button:has-text("Run"):last-child');
  }

  get agentTitle(): Locator {
    return this.page.locator("h1").first();
  }

  async navigateToLibrary(): Promise<void> {
    await this.libraryTab.click();
    await this.page.waitForURL(/.*\/library/);
  }

  async clickFirstAgent(): Promise<void> {
    const firstAgent = this.agentCards.first();
    await firstAgent.click();
  }

  async navigateToAgentByName(agentName: string): Promise<void> {
    const agentCard = this.agentCards.filter({ hasText: agentName }).first();
    await agentCard.click();
  }

  async clickRunButton(): Promise<void> {
    await this.runButton.click();
  }

  async clickNewRunButton(): Promise<void> {
    await this.newRunButton.click();
  }

  async runAgent(inputs: Record<string, string> = {}): Promise<void> {
    await this.clickRunButton();

    // Fill in any required inputs
    for (const [key, value] of Object.entries(inputs)) {
      const input = this.page.locator(
        `input[placeholder*="${key}"], textarea[placeholder*="${key}"]`,
      );
      if (await input.isVisible()) {
        await input.fill(value);
      }
    }

    // Click the run button in the dialog
    await this.runDialogRunButton.click();
  }

  async waitForAgentPageLoad(): Promise<void> {
    await this.page.waitForURL(/.*\/library\/agents\/[^/]+/);
    await this.page.waitForLoadState("networkidle");
  }

  async getAgentName(): Promise<string> {
    return (await this.agentTitle.textContent()) || "";
  }

  async isLoaded(): Promise<boolean> {
    return await this.page.locator("h1").isVisible();
  }

  async waitForRunToComplete(timeout = 30000): Promise<void> {
    // Wait for completion badge or status change
    await this.page.waitForSelector(
      ".bg-green-500, .bg-red-500, .bg-purple-500",
      { timeout },
    );
  }

  async getRunStatus(): Promise<string> {
    // Check for different status indicators
    if (await this.page.locator(".animate-spin").isVisible()) {
      return "running";
    } else if (await this.page.locator(".bg-green-500").isVisible()) {
      return "completed";
    } else if (await this.page.locator(".bg-red-500").isVisible()) {
      return "failed";
    }
    return "unknown";
  }
}
