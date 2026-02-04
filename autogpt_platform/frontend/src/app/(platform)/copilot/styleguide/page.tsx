"use client";

import { ChatMessage } from "@/components/contextual/Chat/components/ChatMessage/ChatMessage";
import type { ChatMessageData } from "@/components/contextual/Chat/components/ChatMessage/useChatMessage";

export default function ChatStyleguidePage() {
  const messages: ChatMessageData[] = [
    {
      type: "message",
      role: "user",
      content: "Hello! Can you help me create an agent?",
    },
    {
      type: "message",
      role: "assistant",
      content:
        "I'd be happy to help you create an agent! Let me search for some relevant information first.",
    },
    {
      type: "tool_call",
      toolId: "tool-1",
      toolName: "search_docs",
      arguments: { query: "agent creation" },
    },
    {
      type: "tool_response",
      toolId: "tool-1",
      toolName: "search_docs",
      result: {
        message: "Found documentation about agent creation",
        results: [
          { title: "Creating Agents", url: "/docs/agents" },
          { title: "Agent Configuration", url: "/docs/config" },
        ],
      },
      success: true,
    },
    {
      type: "tool_call",
      toolId: "tool-2",
      toolName: "run_block",
      arguments: { block_name: "Data Processor" },
    },
    {
      type: "tool_response",
      toolId: "tool-2",
      toolName: "run_block",
      result: {
        type: "agent_output",
        output: "Block executed successfully. Processed 1,234 records.",
      },
      success: true,
    },
    {
      type: "tool_call",
      toolId: "tool-3",
      toolName: "find_agent",
      arguments: { query: "data analysis" },
    },
    {
      type: "tool_call",
      toolId: "tool-streaming",
      toolName: "run_block",
      arguments: { block_name: "Streaming Block" },
    },
    {
      type: "login_needed",
      toolName: "run_agent",
      message: "Please sign in to run this agent",
      sessionId: "session-123",
      agentInfo: {
        graph_id: "graph-123",
        name: "Data Analyzer",
        trigger_type: "manual",
      },
    },
    {
      type: "credentials_needed",
      toolName: "run_agent",
      message: "This agent requires credentials to run",
      agentName: "Data Analyzer",
      credentials: [
        {
          provider: "google",
          providerName: "Google",
          credentialTypes: ["oauth2"],
          title: "Google OAuth",
          scopes: ["read", "write"],
        },
        {
          provider: "openai",
          providerName: "OpenAI",
          credentialTypes: ["api_key"],
          title: "OpenAI API Key",
        },
      ],
    },
    {
      type: "clarification_needed",
      toolName: "create_agent",
      message: "I need some clarification before creating the agent",
      sessionId: "session-123",
      questions: [
        {
          question: "What should the agent be called?",
          keyword: "name",
          example: "e.g., 'Customer Support Bot'",
        },
        {
          question: "What is the main purpose of this agent?",
          keyword: "purpose",
          example: "e.g., 'Handle customer inquiries'",
        },
        {
          question: "What data sources should it access?",
          keyword: "sources",
        },
      ],
    },
    {
      type: "no_results",
      toolName: "find_agent",
      message: "No agents found matching your search",
      suggestions: [
        "Try a different search term",
        "Browse the marketplace",
        "Create a new agent",
      ],
    },
    {
      type: "agent_carousel",
      toolId: "tool-4",
      toolName: "agent_carousel",
      agents: [
        {
          id: "agent-1",
          name: "Customer Support Bot",
          description: "Handles customer inquiries and support tickets",
          version: 1,
        },
        {
          id: "agent-2",
          name: "Data Analyzer",
          description: "Analyzes data and generates reports",
          version: 2,
        },
        {
          id: "agent-3",
          name: "Content Generator",
          description: "Generates blog posts and articles",
          version: 1,
        },
      ],
      totalCount: 15,
    },
    {
      type: "execution_started",
      toolId: "tool-5",
      toolName: "run_agent",
      executionId: "exec-123",
      agentName: "Data Analyzer",
      message: "Agent execution has started",
      libraryAgentLink: "/library/agents/data-analyzer",
    },
    {
      type: "operation_started",
      toolName: "create_agent",
      toolId: "tool-6",
      operationId: "op-123",
      message: "Creating agent... This may take a few minutes.",
    },
    {
      type: "operation_pending",
      toolName: "create_agent",
      toolId: "tool-6",
      operationId: "op-123",
      message: "Agent creation is still in progress...",
    },
    {
      type: "operation_in_progress",
      toolName: "create_agent",
      toolCallId: "tool-6",
      message: "An agent creation is already in progress. Please wait.",
    },
    {
      type: "message",
      role: "assistant",
      content:
        "Here's a summary of what we've accomplished:\n\n- Searched for documentation\n- Found relevant agents\n- Started execution\n\nIs there anything else you'd like to do?",
    },
  ];

  const toolResponseMap = new Map<
    string,
    ChatMessageData & { type: "tool_response" }
  >();
  messages.forEach((msg) => {
    if (msg.type === "tool_response") {
      toolResponseMap.set(msg.toolId, msg);
    }
  });

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-neutral-900">
          Chat Message Styleguide
        </h1>
        <p className="mt-2 text-neutral-600">
          Visual reference for all chat message types and components
        </p>
      </div>

      <div className="space-y-4 rounded-lg border border-neutral-200 bg-neutral-50 p-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className="border-b border-neutral-200 pb-4 last:border-b-0"
          >
            <div className="mb-2 text-xs font-semibold uppercase text-neutral-500">
              {message.type}
              {message.type === "message" && ` (${message.role})`}
            </div>
            <ChatMessage
              message={message}
              messages={messages}
              index={index}
              isStreaming={
                message.type === "tool_call" &&
                message.toolId === "tool-streaming"
              }
              toolResponseMap={toolResponseMap}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
