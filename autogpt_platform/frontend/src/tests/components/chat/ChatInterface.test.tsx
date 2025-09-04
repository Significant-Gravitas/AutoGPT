import React from "react";
import {
  render,
  screen,
  // fireEvent,
  waitFor,
  // within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ChatInterface } from "@/components/chat/ChatInterface";
import {
  MockChatAPI,
  // mockFindAgentStream,
  // mockAuthRequiredStream,
} from "@/tests/mocks/chatApi.mock";
// import BackendAPI from "@/lib/autogpt-server-api";

// Mock Next.js router
const mockPush = jest.fn();
const mockReplace = jest.fn();
jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
    replace: mockReplace,
    back: jest.fn(),
    forward: jest.fn(),
    refresh: jest.fn(),
    prefetch: jest.fn(),
  }),
  usePathname: () => "/marketplace/discover",
  useSearchParams: () => new URLSearchParams(),
}));

// Mock the hooks
jest.mock("@/hooks/useChatSession", () => ({
  useChatSession: jest.fn(),
}));
jest.mock("@/hooks/useChatStream", () => ({
  useChatStream: jest.fn(),
}));
jest.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: jest.fn(),
}));

// Mock BackendAPI
jest.mock("@/lib/autogpt-server-api");

describe("ChatInterface", () => {
  // let mockChatAPI: MockChatAPI;
  let mockSession: any;
  let mockSendMessage: jest.Mock;
  let mockStopStreaming: jest.Mock;

  beforeEach(() => {
    // Clear all mocks
    jest.clearAllMocks();

    // Create mock API
    // mockChatAPI = new MockChatAPI();
    new MockChatAPI();

    // Setup mock session
    mockSession = {
      id: "test-session-123",
      created_at: new Date().toISOString(),
      user_id: "user_123",
    };

    // Mock chat session hook
    const { useChatSession } = jest.requireMock("@/hooks/useChatSession");
    useChatSession.mockReturnValue({
      session: mockSession,
      messages: [],
      isLoading: false,
      error: null,
      createSession: jest.fn(),
      loadSession: jest.fn(),
      refreshSession: jest.fn(),
    });

    // Mock chat stream hook
    mockSendMessage = jest.fn();
    mockStopStreaming = jest.fn();
    const { useChatStream } = jest.requireMock("@/hooks/useChatStream");
    useChatStream.mockReturnValue({
      isStreaming: false,
      sendMessage: mockSendMessage,
      stopStreaming: mockStopStreaming,
    });

    // Mock Supabase hook (no user initially)
    const { useSupabase } = jest.requireMock(
      "@/lib/supabase/hooks/useSupabase",
    );
    useSupabase.mockReturnValue({
      user: null,
      isLoading: false,
    });
  });

  describe("Basic Rendering", () => {
    it("should render the chat interface", () => {
      render(<ChatInterface />);

      expect(
        screen.getByText("AI Agent Discovery Assistant"),
      ).toBeInTheDocument();
      expect(
        screen.getByText(/Chat with me to find and set up/),
      ).toBeInTheDocument();
    });

    it("should show welcome message when no messages", () => {
      render(<ChatInterface />);

      expect(
        screen.getByText(/Hello! I'm here to help you discover/),
      ).toBeInTheDocument();
    });

    it("should render chat input area", () => {
      render(<ChatInterface />);

      expect(
        screen.getByPlaceholderText(/Ask about AI agents/i),
      ).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /send/i })).toBeInTheDocument();
    });
  });

  describe("Message Sending", () => {
    it("should send a message when user types and clicks send", async () => {
      const user = userEvent.setup();
      render(<ChatInterface />);

      const input = screen.getByPlaceholderText(/Ask about AI agents/i);
      const sendButton = screen.getByRole("button", { name: /send/i });

      // Type a message
      await user.type(input, "Find automation agents");
      await user.click(sendButton);

      // Check that sendMessage was called
      expect(mockSendMessage).toHaveBeenCalledWith(
        "test-session-123",
        "Find automation agents",
        expect.any(Function),
      );
    });

    it("should clear input after sending message", async () => {
      const user = userEvent.setup();
      render(<ChatInterface />);

      const input = screen.getByPlaceholderText(
        /Ask about AI agents/i,
      ) as HTMLTextAreaElement;

      await user.type(input, "Test message");
      expect(input.value).toBe("Test message");

      await user.click(screen.getByRole("button", { name: /send/i }));

      await waitFor(() => {
        expect(input.value).toBe("");
      });
    });
  });

  describe("SSE Stream Processing", () => {
    it("should handle text_chunk messages", async () => {
      // Mock streaming response
      mockSendMessage.mockImplementation(
        async (sessionId, message, onChunk) => {
          onChunk({ type: "text_chunk", content: "Hello, I can help you!" });
        },
      );

      render(<ChatInterface />);

      const user = userEvent.setup();
      await user.type(
        screen.getByPlaceholderText(/Ask about AI agents/i),
        "Help me",
      );
      await user.click(screen.getByRole("button", { name: /send/i }));

      await waitFor(() => {
        expect(screen.getByText("Help me")).toBeInTheDocument(); // User message
      });
    });

    it("should handle tool_call messages", async () => {
      mockSendMessage.mockImplementation(
        async (sessionId, message, onChunk) => {
          onChunk({
            type: "tool_call",
            tool_id: "call_123",
            tool_name: "find_agent",
            arguments: { search_query: "automation" },
          });
        },
      );

      render(<ChatInterface />);

      const user = userEvent.setup();
      await user.type(
        screen.getByPlaceholderText(/Ask about AI agents/i),
        "Find agents",
      );
      await user.click(screen.getByRole("button", { name: /send/i }));

      await waitFor(() => {
        expect(screen.getByText("🔍 Search Marketplace")).toBeInTheDocument();
      });
    });

    it("should handle agent carousel in tool_response", async () => {
      mockSendMessage.mockImplementation(
        async (sessionId, message, onChunk) => {
          // Send tool call first
          onChunk({
            type: "tool_call",
            tool_id: "call_123",
            tool_name: "find_agent",
            arguments: { search_query: "automation" },
          });

          // Then send carousel response
          onChunk({
            type: "tool_response",
            tool_id: "call_123",
            tool_name: "find_agent",
            result: {
              type: "agent_carousel",
              query: "automation",
              count: 2,
              agents: [
                {
                  id: "agent1",
                  name: "Test Agent 1",
                  sub_heading: "Test subtitle",
                  description: "Test description",
                  creator: "creator1",
                  rating: 4.5,
                  runs: 100,
                },
                {
                  id: "agent2",
                  name: "Test Agent 2",
                  sub_heading: "Another subtitle",
                  description: "Another description",
                  creator: "creator2",
                  rating: 4.0,
                  runs: 50,
                },
              ],
            },
          });
        },
      );

      render(<ChatInterface />);

      const user = userEvent.setup();
      await user.type(
        screen.getByPlaceholderText(/Ask about AI agents/i),
        "Find agents",
      );
      await user.click(screen.getByRole("button", { name: /send/i }));

      await waitFor(() => {
        expect(screen.getByText("Test Agent 1")).toBeInTheDocument();
        expect(screen.getByText("Test Agent 2")).toBeInTheDocument();
      });
    });
  });

  describe("Authentication Flow", () => {
    it("should show auth prompt when login_needed response received", async () => {
      mockSendMessage.mockImplementation(
        async (sessionId, message, onChunk) => {
          onChunk({
            type: "login_needed",
            message: "Please sign in to continue",
            session_id: "session_123",
            agent_info: {
              agent_id: "agent_123",
              name: "Test Agent",
            },
          });
        },
      );

      render(<ChatInterface />);

      const user = userEvent.setup();
      await user.type(
        screen.getByPlaceholderText(/Ask about AI agents/i),
        "Set up agent",
      );
      await user.click(screen.getByRole("button", { name: /send/i }));

      await waitFor(() => {
        expect(screen.getByText("Authentication Required")).toBeInTheDocument();
        expect(
          screen.getByText("Please sign in to continue"),
        ).toBeInTheDocument();
        expect(
          screen.getByRole("button", { name: /Sign In/i }),
        ).toBeInTheDocument();
        expect(
          screen.getByRole("button", { name: /Create Account/i }),
        ).toBeInTheDocument();
      });
    });

    it("should send login confirmation when user logs in", async () => {
      // First render with auth prompt
      const { rerender } = render(<ChatInterface />);

      // Trigger auth prompt
      mockSendMessage.mockImplementation(
        async (sessionId, message, onChunk) => {
          if (!message.includes("logged in")) {
            onChunk({
              type: "login_needed",
              message: "Please sign in",
              session_id: "session_123",
            });
          }
        },
      );

      const user = userEvent.setup();
      await user.type(
        screen.getByPlaceholderText(/Ask about AI agents/i),
        "Set up agent",
      );
      await user.click(screen.getByRole("button", { name: /send/i }));

      await waitFor(() => {
        expect(screen.getByText("Authentication Required")).toBeInTheDocument();
      });

      // Mock user login
      const { useSupabase } = jest.requireMock(
        "@/lib/supabase/hooks/useSupabase",
      );
      useSupabase.mockReturnValue({
        user: { id: "user_123", email: "test@example.com" },
        isLoading: false,
      });

      // Re-render with logged in user
      rerender(<ChatInterface />);

      // Check that auth prompt is removed and confirmation is sent
      await waitFor(() => {
        expect(
          screen.queryByText("Authentication Required"),
        ).not.toBeInTheDocument();
        expect(mockSendMessage).toHaveBeenCalledWith(
          "test-session-123",
          "I have logged in now",
          expect.any(Function),
        );
      });
    });
  });

  describe("Credentials Flow", () => {
    it("should show credentials setup widget for need_credentials response", async () => {
      mockSendMessage.mockImplementation(
        async (sessionId, message, onChunk) => {
          onChunk({
            type: "tool_response",
            tool_id: "call_789",
            result: {
              type: "need_credentials",
              message: "Configure required credentials",
              agent_id: "agent_123",
              configured_credentials: ["github"],
              missing_credentials: ["openai", "slack"],
              total_required: 3,
            },
          });
        },
      );

      render(<ChatInterface />);

      const user = userEvent.setup();
      await user.type(
        screen.getByPlaceholderText(/Ask about AI agents/i),
        "Check credentials",
      );
      await user.click(screen.getByRole("button", { name: /send/i }));

      await waitFor(() => {
        expect(screen.getByText("Credentials Required")).toBeInTheDocument();
        expect(screen.getByText(/1 of 3 configured/)).toBeInTheDocument();
        expect(screen.getByText("GitHub")).toBeInTheDocument(); // Configured
        expect(screen.getByText("OpenAI")).toBeInTheDocument(); // Missing
        expect(screen.getByText("Slack")).toBeInTheDocument(); // Missing
      });
    });
  });

  describe("Agent Setup Flow", () => {
    it("should show agent setup card for successful setup", async () => {
      mockSendMessage.mockImplementation(
        async (sessionId, message, onChunk) => {
          onChunk({
            type: "tool_response",
            tool_id: "call_setup",
            result: {
              status: "success",
              trigger_type: "schedule",
              name: "Daily Task",
              graph_id: "graph_123",
              graph_version: 1,
              schedule_id: "schedule_456",
              cron: "0 9 * * *",
              timezone: "America/New_York",
              next_run: new Date(Date.now() + 86400000).toISOString(),
              added_to_library: true,
              library_id: "lib_789",
              message: "Successfully scheduled agent",
            },
          });
        },
      );

      render(<ChatInterface />);

      const user = userEvent.setup();
      await user.type(
        screen.getByPlaceholderText(/Ask about AI agents/i),
        "Setup agent",
      );
      await user.click(screen.getByRole("button", { name: /send/i }));

      await waitFor(() => {
        expect(screen.getByText("Agent Setup Complete")).toBeInTheDocument();
        expect(
          screen.getByText("Successfully scheduled agent"),
        ).toBeInTheDocument();
        expect(screen.getByText("Scheduled Execution")).toBeInTheDocument();
        expect(screen.getByText(/0 9 \* \* \*/)).toBeInTheDocument();
        expect(
          screen.getByRole("button", { name: /View in Library/i }),
        ).toBeInTheDocument();
        expect(
          screen.getByRole("button", { name: /View Runs/i }),
        ).toBeInTheDocument();
      });
    });
  });

  describe("Error Handling", () => {
    it("should handle error messages properly", async () => {
      const consoleErrorSpy = jest.spyOn(console, "error").mockImplementation();

      mockSendMessage.mockImplementation(
        async (_sessionId, _message, onChunk) => {
          onChunk({
            type: "error",
            content: "Failed to process request: Invalid agent ID",
          });
        },
      );

      render(<ChatInterface />);

      const user = userEvent.setup();
      await user.type(
        screen.getByPlaceholderText(/Ask about AI agents/i),
        "Invalid request",
      );
      await user.click(screen.getByRole("button", { name: /send/i }));

      await waitFor(() => {
        expect(consoleErrorSpy).toHaveBeenCalledWith(
          "Stream error:",
          "Failed to process request: Invalid agent ID",
        );
      });

      consoleErrorSpy.mockRestore();
    });
  });
});
