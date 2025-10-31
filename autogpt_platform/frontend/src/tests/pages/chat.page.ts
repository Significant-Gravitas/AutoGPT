import { Page, Locator } from "@playwright/test";

export class ChatPage {
  constructor(private page: Page) {}

  async goto(sessionId?: string) {
    await this.page.goto(sessionId ? `/chat?session=${sessionId}` : "/chat");
  }

  // Selectors
  getChatInput(): Locator {
    return this.page.locator('textarea[placeholder*="Type a message"]');
  }

  getSendButton(): Locator {
    return this.page.getByRole("button", { name: /send/i });
  }

  getMessages(): Locator {
    return this.page.locator('[data-testid="chat-message"]');
  }

  getMessageByIndex(index: number): Locator {
    return this.getMessages().nth(index);
  }

  getStreamingMessage(): Locator {
    return this.page.locator('[data-testid="streaming-message"]');
  }

  getNewChatButton(): Locator {
    return this.page.getByRole("button", { name: /new chat/i });
  }

  getQuickActionButton(text: string): Locator {
    return this.page.getByRole("button", { name: new RegExp(text, "i") });
  }

  // Tool-specific message selectors
  getToolCallMessage(): Locator {
    return this.page.locator('[data-testid="tool-call-message"]').first();
  }

  getToolResponseMessage(): Locator {
    return this.page.locator('[data-testid="tool-response-message"]').first();
  }

  getLoginPrompt(): Locator {
    return this.page.getByText("Login Required").first();
  }

  getCredentialsPrompt(): Locator {
    return this.page.getByText("Credentials Required").first();
  }

  getNoResultsMessage(): Locator {
    return this.page.getByText("No Results Found").first();
  }

  getAgentCarouselMessage(): Locator {
    return this.page.getByText(/Found \d+ Agents?/).first();
  }

  getExecutionStartedMessage(): Locator {
    return this.page.getByText("Execution Started").first();
  }

  // Actions
  async sendMessage(text: string): Promise<void> {
    const input = this.getChatInput();
    await input.waitFor({ state: "visible" });
    await input.fill(text);

    const sendButton = this.getSendButton();
    await sendButton.waitFor({ state: "visible" });
    await sendButton.click();
  }

  async clickQuickAction(text: string): Promise<void> {
    const button = this.getQuickActionButton(text);
    await button.waitFor({ state: "visible" });
    await button.click();
  }

  async startNewChat(): Promise<void> {
    const button = this.getNewChatButton();
    await button.waitFor({ state: "visible" });
    await button.click();
  }

  async waitForResponse(): Promise<void> {
    // Wait for a new assistant message to appear
    await this.page.waitForSelector('[data-testid="chat-message"]', {
      state: "visible",
      timeout: 10000,
    });
  }

  async waitForStreaming(): Promise<void> {
    await this.getStreamingMessage().waitFor({
      state: "visible",
      timeout: 5000,
    });
  }

  async waitForToolCall(): Promise<void> {
    await this.getToolCallMessage().waitFor({
      state: "visible",
      timeout: 10000,
    });
  }

  async waitForToolResponse(): Promise<void> {
    await this.getToolResponseMessage().waitFor({
      state: "visible",
      timeout: 10000,
    });
  }

  async waitForLoginPrompt(): Promise<void> {
    await this.getLoginPrompt().waitFor({
      state: "visible",
      timeout: 5000,
    });
  }

  async waitForCredentialsPrompt(): Promise<void> {
    await this.getCredentialsPrompt().waitFor({
      state: "visible",
      timeout: 5000,
    });
  }

  async waitForNoResults(): Promise<void> {
    await this.getNoResultsMessage().waitFor({
      state: "visible",
      timeout: 5000,
    });
  }

  async waitForAgentCarousel(): Promise<void> {
    await this.getAgentCarouselMessage().waitFor({
      state: "visible",
      timeout: 5000,
    });
  }

  async waitForExecutionStarted(): Promise<void> {
    await this.getExecutionStartedMessage().waitFor({
      state: "visible",
      timeout: 5000,
    });
  }
}
