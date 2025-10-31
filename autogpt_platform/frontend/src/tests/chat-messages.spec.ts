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
    // Mock session GET using URL param session=test-session
    await page.route("**/api/chat/sessions/test-session", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: "test-session",
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          user_id: null,
          messages: [],
        }),
      });
    });

    // Mock SSE stream to emit a tool_call event
    await page.route(
      "**/api/chat/sessions/test-session/stream?**",
      async (route) => {
        const sse = [
          `data: ${JSON.stringify({
            type: "tool_call",
            tool_id: "tool-123",
            tool_name: "find_agent",
            arguments: { query: "data analysis" },
          })}\n\n`,
          `data: ${JSON.stringify({ type: "stream_end" })}\n\n`,
        ].join("");
        await route.fulfill({
          status: 200,
          contentType: "text/event-stream",
          body: sse,
        });
      },
    );

    const chatPage = new ChatPage(page);
    await chatPage.goto("test-session");

    // Trigger streaming
    await chatPage.sendMessage("Find data analysis agents");

    const toolCallText = page.getByText("find_agent");
    await expect(toolCallText).toBeVisible({ timeout: 5000 });
  });
});

test.describe("Chat Messages - Tool Response Message", () => {
  test("displays successful tool response in green", async ({ page }) => {
    await page.route("**/api/chat/sessions/test-session", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: "test-session",
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          user_id: null,
          messages: [],
        }),
      });
    });

    await page.route(
      "**/api/chat/sessions/test-session/stream?**",
      async (route) => {
        const sse = [
          `data: ${JSON.stringify({
            type: "tool_response",
            tool_id: "tool-456",
            tool_name: "get_agent_details",
            result: { name: "Test Agent", version: 1 },
            success: true,
          })}\n\n`,
          `data: ${JSON.stringify({ type: "stream_end" })}\n\n`,
        ].join("");
        await route.fulfill({
          status: 200,
          contentType: "text/event-stream",
          body: sse,
        });
      },
    );

    const chatPage = new ChatPage(page);
    await chatPage.goto("test-session");
    await chatPage.sendMessage("details");

    const completedText = page.getByText(/Completed.*get_agent_details/);
    await expect(completedText).toBeVisible({ timeout: 5000 });
  });

  test("displays failed tool response in red", async ({ page }) => {
    await page.route("**/api/chat/sessions/test-session", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: "test-session",
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          user_id: null,
          messages: [],
        }),
      });
    });

    await page.route(
      "**/api/chat/sessions/test-session/stream?**",
      async (route) => {
        const sse = [
          `data: ${JSON.stringify({
            type: "tool_response",
            tool_id: "tool-789",
            tool_name: "run_agent",
            result: "Error: Agent not found",
            success: false,
          })}\n\n`,
          `data: ${JSON.stringify({ type: "stream_end" })}\n\n`,
        ].join("");
        await route.fulfill({
          status: 200,
          contentType: "text/event-stream",
          body: sse,
        });
      },
    );

    const chatPage = new ChatPage(page);
    await chatPage.goto("test-session");
    await chatPage.sendMessage("run");

    const failedText = page.getByText(/Failed.*run_agent/);
    await expect(failedText).toBeVisible({ timeout: 5000 });
  });
});

test.describe("Chat Messages - Login Prompt", () => {
  test("displays login prompt with action buttons", async ({ page }) => {
    await page.route("**/api/chat/sessions/test-session", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: "test-session",
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          user_id: null,
          messages: [],
        }),
      });
    });

    await page.route(
      "**/api/chat/sessions/test-session/stream?**",
      async (route) => {
        const sse = [
          `data: ${JSON.stringify({
            type: "login_needed",
            message:
              "To run agents and save your chat history, please log in to your account.",
            session_id: "test-session",
          })}\n\n`,
          `data: ${JSON.stringify({ type: "stream_end" })}\n\n`,
        ].join("");
        await route.fulfill({
          status: 200,
          contentType: "text/event-stream",
          body: sse,
        });
      },
    );

    const chatPage = new ChatPage(page);
    await chatPage.goto("test-session");
    await chatPage.sendMessage("login");

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
    await page.route("**/api/chat/sessions/test-session", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: "test-session",
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          user_id: null,
          messages: [],
        }),
      });
    });

    await page.route(
      "**/api/chat/sessions/test-session/stream?**",
      async (route) => {
        const setupInfo = {
          setup_info: {
            agent_name: "GitHub Integration",
            user_readiness: {
              missing_credentials: {
                github: {
                  provider: "github",
                  provider_name: "GitHub",
                  type: "oauth2",
                  scopes: [],
                },
              },
            },
          },
        };
        const sse = [
          `data: ${JSON.stringify({
            type: "tool_response",
            tool_id: "tool-setup",
            tool_name: "get_required_setup_info",
            result: setupInfo,
            success: true,
          })}\n\n`,
          `data: ${JSON.stringify({ type: "stream_end" })}\n\n`,
        ].join("");
        await route.fulfill({
          status: 200,
          contentType: "text/event-stream",
          body: sse,
        });
      },
    );

    const chatPage = new ChatPage(page);
    await chatPage.goto("test-session");
    await chatPage.sendMessage("run github agent");

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
    await page.route("**/api/chat/sessions/test-session", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: "test-session",
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          user_id: null,
          messages: [],
        }),
      });
    });

    await page.route(
      "**/api/chat/sessions/test-session/stream?**",
      async (route) => {
        const result = {
          type: "no_results",
          message:
            "No agents found matching 'crypto mining'. Try different keywords or browse the marketplace.",
          suggestions: [
            "Try more general terms",
            "Browse categories in the marketplace",
            "Check spelling",
          ],
        };
        const sse = [
          `data: ${JSON.stringify({
            type: "tool_response",
            tool_id: "tool-find",
            tool_name: "find_agent",
            result,
            success: true,
          })}\n\n`,
          `data: ${JSON.stringify({ type: "stream_end" })}\n\n`,
        ].join("");
        await route.fulfill({
          status: 200,
          contentType: "text/event-stream",
          body: sse,
        });
      },
    );

    const chatPage = new ChatPage(page);
    await chatPage.goto("test-session");
    await chatPage.sendMessage("crypto mining");

    const noResultsHeader = page.getByText("No Results Found");
    await expect(noResultsHeader).toBeVisible({ timeout: 5000 });

    const suggestion = page.getByText("Try more general terms");
    await expect(suggestion).toBeVisible();
  });
});

test.describe("Chat Messages - Agent Carousel Message", () => {
  test("displays agent carousel with multiple agents", async ({ page }) => {
    await page.route("**/api/chat/sessions/test-session", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: "test-session",
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          user_id: null,
          messages: [],
        }),
      });
    });

    await page.route(
      "**/api/chat/sessions/test-session/stream?**",
      async (route) => {
        const result = {
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
          total_count: 8,
        };
        const sse = [
          `data: ${JSON.stringify({
            type: "tool_response",
            tool_id: "tool-find",
            tool_name: "find_agent",
            result,
            success: true,
          })}\n\n`,
          `data: ${JSON.stringify({ type: "stream_end" })}\n\n`,
        ].join("");
        await route.fulfill({
          status: 200,
          contentType: "text/event-stream",
          body: sse,
        });
      },
    );

    const chatPage = new ChatPage(page);
    await chatPage.goto("test-session");
    await chatPage.sendMessage("find agents");

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
    await page.route("**/api/chat/sessions/test-session", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: "test-session",
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          user_id: null,
          messages: [],
        }),
      });
    });

    await page.route(
      "**/api/chat/sessions/test-session/stream?**",
      async (route) => {
        const result = {
          type: "execution_started",
          execution_id: "exec-123e4567-e89b-12d3-a456-426614174000",
          agent_name: "Data Analysis Agent",
          message: "Your agent execution has started successfully",
        };
        const sse = [
          `data: ${JSON.stringify({
            type: "tool_response",
            tool_id: "tool-run",
            tool_name: "run_agent",
            result,
            success: true,
          })}\n\n`,
          `data: ${JSON.stringify({ type: "stream_end" })}\n\n`,
        ].join("");
        await route.fulfill({
          status: 200,
          contentType: "text/event-stream",
          body: sse,
        });
      },
    );

    const chatPage = new ChatPage(page);
    await chatPage.goto("test-session");
    await chatPage.sendMessage("run agent");

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
    await page.route("**/api/chat/sessions/test-session", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: "test-session",
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          user_id: null,
          messages: [],
        }),
      });
    });

    await page.route(
      "**/api/chat/sessions/test-session/stream?**",
      async (route) => {
        const sse = [
          // tool call
          `data: ${JSON.stringify({
            type: "tool_call",
            tool_id: "tool-111",
            tool_name: "find_agent",
            arguments: { query: "automation" },
          })}\n\n`,
          // agent carousel
          `data: ${JSON.stringify({
            type: "tool_response",
            tool_id: "tool-111",
            tool_name: "find_agent",
            result: {
              type: "agent_carousel",
              agents: [
                {
                  id: "agent-1",
                  name: "Email Automation",
                  description: "Automates emails",
                  version: 1,
                },
              ],
              total_count: 5,
            },
            success: true,
          })}\n\n`,
          // credentials needed via setup info
          `data: ${JSON.stringify({
            type: "tool_response",
            tool_id: "tool-setup",
            tool_name: "get_required_setup_info",
            result: {
              setup_info: {
                agent_name: "Email Automation",
                user_readiness: {
                  missing_credentials: {
                    gmail: {
                      provider: "gmail",
                      provider_name: "Gmail",
                      type: "oauth2",
                      scopes: [],
                    },
                  },
                },
              },
            },
            success: true,
          })}\n\n`,
          `data: ${JSON.stringify({ type: "stream_end" })}\n\n`,
        ].join("");
        await route.fulfill({
          status: 200,
          contentType: "text/event-stream",
          body: sse,
        });
      },
    );

    const chatPage = new ChatPage(page);
    await chatPage.goto("test-session");
    await chatPage.sendMessage("Find automation agents");

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
