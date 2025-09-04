/**
 * Mock implementation of the Chat API for testing
 */

import {
  ChatSession,
  ChatMessage,
  StreamChunk,
} from "@/lib/autogpt-server-api/chat";

export class MockEventSource {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  readyState: number = 0;
  url: string;

  constructor(url: string) {
    this.url = url;
    this.readyState = 1; // OPEN
  }

  close() {
    this.readyState = 2; // CLOSED
  }
}

export const mockSessions: Map<string, ChatSession> = new Map();
export const mockMessages: Map<string, ChatMessage[]> = new Map();

// Helper to generate SSE event
export function createSSEEvent(data: any): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

// Mock stream generators for different scenarios
export async function* mockFindAgentStream(): AsyncGenerator<StreamChunk> {
  // Initial text
  yield {
    type: "text_chunk",
    content: "I'll search for agents that can help you. ",
  } as StreamChunk;

  await delay(100);

  // Tool call
  yield {
    type: "tool_call",
    tool_id: "call_123",
    tool_name: "find_agent",
    arguments: { search_query: "automation" },
  } as StreamChunk;

  await delay(200);

  // Tool response with carousel
  yield {
    type: "tool_response",
    tool_id: "call_123",
    tool_name: "find_agent",
    result: {
      type: "agent_carousel",
      query: "automation",
      count: 3,
      agents: [
        {
          id: "user/email-automation",
          name: "Email Automation Agent",
          sub_heading: "Automate your email workflows",
          description: "Automatically process and respond to emails",
          creator: "john_doe",
          creator_avatar: "/avatar1.png",
          agent_image: "/agent1.png",
          rating: 4.5,
          runs: 1523,
        },
        {
          id: "user/web-scraper",
          name: "Web Scraper Agent",
          sub_heading: "Extract data from websites",
          description: "Scrape and monitor websites for changes",
          creator: "jane_smith",
          creator_avatar: "/avatar2.png",
          agent_image: "/agent2.png",
          rating: 4.8,
          runs: 3421,
        },
        {
          id: "user/slack-bot",
          name: "Slack Integration Bot",
          sub_heading: "Connect Slack to your workflows",
          description: "Automate Slack messages and responses",
          creator: "bot_builder",
          creator_avatar: "/avatar3.png",
          agent_image: "/agent3.png",
          rating: 4.2,
          runs: 892,
        },
      ],
    },
  } as StreamChunk;

  await delay(100);

  // Follow-up text
  yield {
    type: "text_chunk",
    content:
      "\n\nI found 3 automation agents that might help you. Each one specializes in different types of automation tasks.",
  } as StreamChunk;

  // End stream
  yield {
    type: "stream_end",
    content: "",
    summary: { message_count: 2, had_tool_calls: true },
  } as StreamChunk;
}

export async function* mockAuthRequiredStream(): AsyncGenerator<StreamChunk> {
  yield {
    type: "text_chunk",
    content: "Let me get the details for this agent. ",
  } as StreamChunk;

  await delay(100);

  // Tool call
  yield {
    type: "tool_call",
    tool_id: "call_456",
    tool_name: "get_agent_details",
    arguments: { agent_id: "user/email-automation", agent_version: "1" },
  } as StreamChunk;

  await delay(200);

  // Login needed response
  yield {
    type: "login_needed",
    message:
      "This agent requires credentials. Please sign in to set up and use this agent.",
    session_id: "session_123",
    agent_info: {
      agent_id: "user/email-automation",
      agent_version: "1",
      name: "Email Automation Agent",
      graph_id: "graph_123",
    },
  } as StreamChunk;

  yield {
    type: "stream_end",
    content: "",
  } as StreamChunk;
}

export async function* mockCredentialsNeededStream(): AsyncGenerator<StreamChunk> {
  yield {
    type: "text_chunk",
    content: "Checking what credentials are needed for this agent... ",
  } as StreamChunk;

  await delay(100);

  yield {
    type: "tool_call",
    tool_id: "call_789",
    tool_name: "check_credentials",
    arguments: {
      agent_id: "agent_123",
      required_credentials: ["github", "openai", "slack"],
    },
  } as StreamChunk;

  await delay(200);

  yield {
    type: "tool_response",
    tool_id: "call_789",
    tool_name: "check_credentials",
    result: {
      type: "need_credentials",
      message: "Some credentials need to be configured",
      agent_id: "agent_123",
      configured_credentials: ["github"],
      missing_credentials: ["openai", "slack"],
      total_required: 3,
    },
  } as StreamChunk;

  yield {
    type: "stream_end",
    content: "",
  } as StreamChunk;
}

export async function* mockSetupAgentStream(): AsyncGenerator<StreamChunk> {
  yield {
    type: "text_chunk",
    content: "Setting up your agent with a daily schedule... ",
  } as StreamChunk;

  await delay(100);

  yield {
    type: "tool_call",
    tool_id: "call_setup",
    tool_name: "setup_agent",
    arguments: {
      graph_id: "graph_123",
      graph_version: 1,
      name: "Daily Email Processor",
      trigger_type: "schedule",
      cron: "0 9 * * *",
      inputs: { mailbox: "inbox" },
    },
  } as StreamChunk;

  await delay(300);

  yield {
    type: "tool_response",
    tool_id: "call_setup",
    tool_name: "setup_agent",
    result: {
      status: "success",
      trigger_type: "schedule",
      name: "Daily Email Processor",
      graph_id: "graph_123",
      graph_version: 1,
      schedule_id: "schedule_456",
      cron: "0 9 * * *",
      cron_utc: "0 14 * * *",
      timezone: "America/New_York",
      next_run: new Date(Date.now() + 86400000).toISOString(),
      added_to_library: true,
      library_id: "lib_789",
      message:
        "Successfully scheduled 'Email Automation Agent' to run daily at 9:00 AM",
    },
  } as StreamChunk;

  yield {
    type: "text_chunk",
    content:
      "\n\nYour agent has been successfully set up! It will run every day at 9:00 AM.",
  } as StreamChunk;

  yield {
    type: "stream_end",
    content: "",
  } as StreamChunk;
}

// Helper delay function
function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Mock ChatAPI class
export class MockChatAPI {
  async createSession(request?: any): Promise<ChatSession> {
    const session: ChatSession = {
      id: `session_${Date.now()}`,
      created_at: new Date().toISOString(),
      user_id: request?.metadata?.anon_id || "user_123",
      messages: [],
      metadata: request?.metadata || {},
    };

    mockSessions.set(session.id, session);
    mockMessages.set(session.id, []);

    if (request?.system_prompt) {
      mockMessages.get(session.id)!.push({
        content: request.system_prompt,
        role: "SYSTEM",
        created_at: new Date().toISOString(),
      });
    }

    return session;
  }

  async getSession(sessionId: string): Promise<ChatSession> {
    const session = mockSessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    return {
      ...session,
      messages: mockMessages.get(sessionId) || [],
    };
  }

  async *streamChat(
    sessionId: string,
    message: string,
    _model = "gpt-4o",
    _maxContext = 50,
  ): AsyncGenerator<StreamChunk> {
    // Add user message
    const userMessage: ChatMessage = {
      content: message,
      role: "USER",
      created_at: new Date().toISOString(),
    };

    const messages = mockMessages.get(sessionId) || [];
    messages.push(userMessage);
    mockMessages.set(sessionId, messages);

    // Choose response based on message content
    if (
      message.toLowerCase().includes("find") ||
      message.toLowerCase().includes("search")
    ) {
      yield* mockFindAgentStream();
    } else if (
      message.toLowerCase().includes("set up") &&
      !message.includes("logged in")
    ) {
      yield* mockAuthRequiredStream();
    } else if (message.toLowerCase().includes("credentials")) {
      yield* mockCredentialsNeededStream();
    } else if (
      message.toLowerCase().includes("schedule") ||
      message.toLowerCase().includes("logged in")
    ) {
      yield* mockSetupAgentStream();
    } else {
      // Default response
      yield {
        type: "text_chunk",
        content:
          "I can help you find and set up AI agents. Try asking me to search for specific types of agents!",
      } as StreamChunk;

      yield {
        type: "stream_end",
        content: "",
      } as StreamChunk;
    }

    // Store assistant message
    const assistantMessage: ChatMessage = {
      content: "Response generated",
      role: "ASSISTANT",
      created_at: new Date().toISOString(),
    };
    messages.push(assistantMessage);
    mockMessages.set(sessionId, messages);
  }
}

// Export mock factory
export function createMockChatAPI() {
  return new MockChatAPI();
}
