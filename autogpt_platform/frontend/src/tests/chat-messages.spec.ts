import { test, expect } from "@playwright/test";
import { ChatPage } from "./pages/chat.page";

/**
 * Chat Messages Component Tests
 *
 * These tests verify that different chat message types render correctly
 * using mock data instead of calling the backend.
 */

test.describe("Chat Messages - Tool Call Message", () => {
  test("displays tool call message with spinning animation", async ({
    page,
  }) => {
    // Mock the session API to return a session with a tool call message
    await page.route("**/api/v1/chat/sessions/*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          session_id: "test-session",
          messages: [],
        }),
      });
    });

    // Navigate to chat and inject messages via page.evaluate
    const chatPage = new ChatPage(page);
    await chatPage.goto();

    // Inject messages directly into the component state for testing
    await page.evaluate(() => {
      const mockMessages = [
        {
          type: "message",
          role: "user",
          content: "Find data analysis agents",
          timestamp: new Date(),
        },
        {
          type: "tool_call",
          toolId: "tool-123",
          toolName: "find_agent",
          arguments: { query: "data analysis" },
          timestamp: new Date(),
        },
      ];

      // Dispatch a custom event or use React DevTools to inject state
      window.dispatchEvent(
        new CustomEvent("test:inject-messages", {
          detail: { messages: mockMessages },
        }),
      );
    });

    // Verify tool call message is visible
    const toolCallText = page.getByText("find_agent");
    await expect(toolCallText).toBeVisible({ timeout: 5000 });
  });
});

test.describe("Chat Messages - Tool Response Message", () => {
  test("displays successful tool response in green", async ({ page }) => {
    await page.route("**/api/v1/chat/sessions/*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          session_id: "test-session",
          messages: [],
        }),
      });
    });

    const chatPage = new ChatPage(page);
    await chatPage.goto();

    await page.evaluate(() => {
      const mockMessages = [
        {
          type: "tool_response",
          toolId: "tool-456",
          toolName: "get_agent_details",
          result: { name: "Test Agent", version: 1 },
          success: true,
          timestamp: new Date(),
        },
      ];

      window.dispatchEvent(
        new CustomEvent("test:inject-messages", {
          detail: { messages: mockMessages },
        }),
      );
    });

    const completedText = page.getByText(/Completed.*get_agent_details/);
    await expect(completedText).toBeVisible({ timeout: 5000 });
  });

  test("displays failed tool response in red", async ({ page }) => {
    await page.route("**/api/v1/chat/sessions/*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          session_id: "test-session",
          messages: [],
        }),
      });
    });

    const chatPage = new ChatPage(page);
    await chatPage.goto();

    await page.evaluate(() => {
      const mockMessages = [
        {
          type: "tool_response",
          toolId: "tool-789",
          toolName: "run_agent",
          result: "Error: Agent not found",
          success: false,
          timestamp: new Date(),
        },
      ];

      window.dispatchEvent(
        new CustomEvent("test:inject-messages", {
          detail: { messages: mockMessages },
        }),
      );
    });

    const failedText = page.getByText(/Failed.*run_agent/);
    await expect(failedText).toBeVisible({ timeout: 5000 });
  });
});

test.describe("Chat Messages - Login Prompt", () => {
  test("displays login prompt with action buttons", async ({ page }) => {
    await page.route("**/api/v1/chat/sessions/*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          session_id: "test-session",
          messages: [],
        }),
      });
    });

    const chatPage = new ChatPage(page);
    await chatPage.goto();

    await page.evaluate(() => {
      const mockMessages = [
        {
          type: "login_needed",
          message:
            "To run agents and save your chat history, please log in to your account.",
          sessionId: "test-session",
          timestamp: new Date(),
        },
      ];

      window.dispatchEvent(
        new CustomEvent("test:inject-messages", {
          detail: { messages: mockMessages },
        }),
      );
    });

    // Check for login prompt elements
    const loginHeader = page.getByText("Login Required");
    await expect(loginHeader).toBeVisible({ timeout: 5000 });

    const loginButton = page.getByRole("button", { name: /login/i });
    await expect(loginButton).toBeVisible();

    const guestButton = page.getByRole("button", {
      name: /continue as guest/i,
    });
    await expect(guestButton).toBeVisible();
  });
});

test.describe("Chat Messages - Credentials Prompt", () => {
  test("displays credentials prompt for OAuth", async ({ page }) => {
    await page.route("**/api/v1/chat/sessions/*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          session_id: "test-session",
          messages: [],
        }),
      });
    });

    const chatPage = new ChatPage(page);
    await chatPage.goto();

    await page.evaluate(() => {
      const mockMessages = [
        {
          type: "credentials_needed",
          provider: "github",
          providerName: "GitHub",
          credentialType: "oauth2",
          title: "GitHub Integration",
          message:
            "To run the GitHub Integration agent, you need to add GitHub credentials.",
          timestamp: new Date(),
        },
      ];

      window.dispatchEvent(
        new CustomEvent("test:inject-messages", {
          detail: { messages: mockMessages },
        }),
      );
    });

    // Check for credentials prompt elements
    const credentialsHeader = page.getByText("Credentials Required");
    await expect(credentialsHeader).toBeVisible({ timeout: 5000 });

    const setupButton = page.getByRole("button", {
      name: /setup credentials/i,
    });
    await expect(setupButton).toBeVisible();

    const providerName = page.getByText("GitHub");
    await expect(providerName).toBeVisible();
  });
});

test.describe("Chat Messages - No Results Message", () => {
  test("displays no results message with suggestions", async ({ page }) => {
    await page.route("**/api/v1/chat/sessions/*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          session_id: "test-session",
          messages: [],
        }),
      });
    });

    const chatPage = new ChatPage(page);
    await chatPage.goto();

    await page.evaluate(() => {
      const mockMessages = [
        {
          type: "no_results",
          message:
            "No agents found matching 'crypto mining'. Try different keywords or browse the marketplace.",
          suggestions: [
            "Try more general terms",
            "Browse categories in the marketplace",
            "Check spelling",
          ],
          timestamp: new Date(),
        },
      ];

      window.dispatchEvent(
        new CustomEvent("test:inject-messages", {
          detail: { messages: mockMessages },
        }),
      );
    });

    // Check for no results elements
    const noResultsHeader = page.getByText("No Results Found");
    await expect(noResultsHeader).toBeVisible({ timeout: 5000 });

    const suggestion = page.getByText("Try more general terms");
    await expect(suggestion).toBeVisible();
  });
});

test.describe("Chat Messages - Agent Carousel Message", () => {
  test("displays agent carousel with multiple agents", async ({ page }) => {
    await page.route("**/api/v1/chat/sessions/*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          session_id: "test-session",
          messages: [],
        }),
      });
    });

    const chatPage = new ChatPage(page);
    await chatPage.goto();

    await page.evaluate(() => {
      const mockMessages = [
        {
          type: "agent_carousel",
          agents: [
            {
              id: "agent-1",
              name: "Email Automation",
              description: "Automates email responses",
              version: 1,
            },
            {
              id: "agent-2",
              name: "Social Media Manager",
              description: "Schedules social posts",
              version: 2,
            },
          ],
          totalCount: 8,
          timestamp: new Date(),
        },
      ];

      window.dispatchEvent(
        new CustomEvent("test:inject-messages", {
          detail: { messages: mockMessages },
        }),
      );
    });

    // Check for agent carousel elements
    const carouselHeader = page.getByText(/Found \d+ Agents/);
    await expect(carouselHeader).toBeVisible({ timeout: 5000 });

    const agent1 = page.getByText("Email Automation");
    await expect(agent1).toBeVisible();

    const agent2 = page.getByText("Social Media Manager");
    await expect(agent2).toBeVisible();
  });
});

test.describe("Chat Messages - Execution Started Message", () => {
  test("displays execution started confirmation", async ({ page }) => {
    await page.route("**/api/v1/chat/sessions/*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          session_id: "test-session",
          messages: [],
        }),
      });
    });

    const chatPage = new ChatPage(page);
    await chatPage.goto();

    await page.evaluate(() => {
      const mockMessages = [
        {
          type: "execution_started",
          executionId: "exec-123e4567-e89b-12d3-a456-426614174000",
          agentName: "Data Analysis Agent",
          message: "Your agent execution has started successfully",
          timestamp: new Date(),
        },
      ];

      window.dispatchEvent(
        new CustomEvent("test:inject-messages", {
          detail: { messages: mockMessages },
        }),
      );
    });

    // Check for execution started elements
    const executionHeader = page.getByText("Execution Started");
    await expect(executionHeader).toBeVisible({ timeout: 5000 });

    const agentName = page.getByText("Data Analysis Agent");
    await expect(agentName).toBeVisible();

    const executionId = page.getByText(/exec-123e4567/);
    await expect(executionId).toBeVisible();
  });
});

test.describe("Chat Messages - Mixed Conversation", () => {
  test("displays multiple message types in sequence", async ({ page }) => {
    await page.route("**/api/v1/chat/sessions/*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          session_id: "test-session",
          messages: [],
        }),
      });
    });

    const chatPage = new ChatPage(page);
    await chatPage.goto();

    await page.evaluate(() => {
      const mockMessages = [
        {
          type: "message",
          role: "user",
          content: "Find automation agents",
          timestamp: new Date(Date.now() - 5 * 60 * 1000),
        },
        {
          type: "tool_call",
          toolId: "tool-111",
          toolName: "find_agent",
          arguments: { query: "automation" },
          timestamp: new Date(Date.now() - 4 * 60 * 1000),
        },
        {
          type: "agent_carousel",
          agents: [
            {
              id: "agent-1",
              name: "Email Automation",
              description: "Automates emails",
              version: 1,
            },
          ],
          totalCount: 5,
          timestamp: new Date(Date.now() - 3 * 60 * 1000),
        },
        {
          type: "message",
          role: "user",
          content: "Run the first one",
          timestamp: new Date(Date.now() - 2 * 60 * 1000),
        },
        {
          type: "credentials_needed",
          provider: "gmail",
          providerName: "Gmail",
          credentialType: "oauth2",
          title: "Email Automation",
          message: "You need Gmail credentials",
          timestamp: new Date(Date.now() - 1 * 60 * 1000),
        },
      ];

      window.dispatchEvent(
        new CustomEvent("test:inject-messages", {
          detail: { messages: mockMessages },
        }),
      );
    });

    // Verify multiple message types are present
    const userMessage = page.getByText("Find automation agents");
    await expect(userMessage).toBeVisible({ timeout: 5000 });

    const toolCall = page.getByText("find_agent");
    await expect(toolCall).toBeVisible();

    const carousel = page.getByText(/Found \d+ Agents/);
    await expect(carousel).toBeVisible();

    const credentials = page.getByText("Credentials Required");
    await expect(credentials).toBeVisible();
  });
});
