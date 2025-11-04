import type { Meta, StoryObj } from "@storybook/nextjs";
import { MessageList } from "./MessageList";
import { useEffect, useState } from "react";

const meta = {
  title: "Molecules/MessageList",
  component: MessageList,
  parameters: {
    layout: "fullscreen",
  },
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="h-screen p-4">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof MessageList>;

export default meta;
type Story = StoryObj<typeof meta>;

const sampleMessages = [
  {
    type: "message" as const,
    role: "user" as const,
    content: "Hello! How can you help me today?",
    timestamp: new Date(Date.now() - 10 * 60 * 1000),
  },
  {
    type: "message" as const,
    role: "assistant" as const,
    content:
      "I can help you discover and run AI agents! What would you like to do?",
    timestamp: new Date(Date.now() - 9 * 60 * 1000),
  },
  {
    type: "message" as const,
    role: "user" as const,
    content: "Find me automation agents",
    timestamp: new Date(Date.now() - 8 * 60 * 1000),
  },
  {
    type: "message" as const,
    role: "assistant" as const,
    content: "I found 15 automation agents. Here are the top 3...",
    timestamp: new Date(Date.now() - 7 * 60 * 1000),
  },
];

export const Empty: Story = {
  args: {
    messages: [],
  },
};

export const FewMessages: Story = {
  args: {
    messages: sampleMessages.slice(0, 2),
  },
};

export const ManyMessages: Story = {
  args: {
    messages: sampleMessages,
  },
};

export const WithStreaming: Story = {
  args: {
    messages: sampleMessages,
    isStreaming: true,
    streamingChunks: [
      "Let me ",
      "help you ",
      "with that. ",
      "I can show you ",
      "the details...",
    ],
  },
};

export const SimulatedConversation: Story = {
  args: {
    messages: [],
  },
  render: () => {
    const [messages, setMessages] = useState(sampleMessages);
    const [streamingChunks, setStreamingChunks] = useState<string[]>([]);
    const [isStreaming, setIsStreaming] = useState(false);

    useEffect(function simulateConversation() {
      const timer = setTimeout(() => {
        setIsStreaming(true);
        const fullText =
          "This is a simulated streaming response that demonstrates the auto-scroll behavior and real-time message updates!";
        const words = fullText.split(" ");
        let index = 0;

        const interval = setInterval(() => {
          if (index < words.length) {
            setStreamingChunks((prev) => [...prev, words[index] + " "]);
            index++;
          } else {
            clearInterval(interval);
            setTimeout(() => {
              setIsStreaming(false);
              setMessages((prev) => [
                ...prev,
                {
                  type: "message",
                  role: "assistant",
                  content: fullText,
                  timestamp: new Date(),
                },
              ]);
              setStreamingChunks([]);
            }, 500);
          }
        }, 100);
      }, 1000);

      return () => clearTimeout(timer);
    }, []);

    return (
      <MessageList
        messages={messages}
        streamingChunks={streamingChunks}
        isStreaming={isStreaming}
      />
    );
  },
};

export const LongConversation: Story = {
  args: {
    messages: [
      ...sampleMessages,
      ...sampleMessages.map((msg, i) => ({
        ...msg,
        timestamp: new Date(Date.now() - (6 - i) * 60 * 1000),
      })),
      ...sampleMessages.map((msg, i) => ({
        ...msg,
        timestamp: new Date(Date.now() - (2 - i) * 60 * 1000),
      })),
    ],
  },
};

export const WithToolCalls: Story = {
  args: {
    messages: [
      {
        type: "message" as const,
        role: "user" as const,
        content: "Find me data analysis agents",
        timestamp: new Date(Date.now() - 5 * 60 * 1000),
      },
      {
        type: "tool_call" as const,
        toolId: "tool-123e4567-e89b-12d3-a456-426614174000",
        toolName: "find_agent",
        arguments: { query: "data analysis" },
        timestamp: new Date(Date.now() - 4 * 60 * 1000),
      },
      {
        type: "message" as const,
        role: "assistant" as const,
        content: "I found several data analysis agents for you!",
        timestamp: new Date(Date.now() - 3 * 60 * 1000),
      },
    ],
  },
};

export const WithToolResponses: Story = {
  args: {
    messages: [
      {
        type: "message" as const,
        role: "user" as const,
        content: "Get details about this agent",
        timestamp: new Date(Date.now() - 5 * 60 * 1000),
      },
      {
        type: "tool_call" as const,
        toolId: "tool-456a7890-b12c-34d5-e678-901234567def",
        toolName: "get_agent_details",
        arguments: { agent_id: "agent-123" },
        timestamp: new Date(Date.now() - 4 * 60 * 1000),
      },
      {
        type: "tool_response" as const,
        toolId: "tool-456a7890-b12c-34d5-e678-901234567def",
        toolName: "get_agent_details",
        result: {
          name: "Data Analysis Agent",
          description: "Analyzes CSV and Excel files",
          version: 1,
        },
        success: true,
        timestamp: new Date(Date.now() - 3 * 60 * 1000),
      },
    ],
  },
};

export const WithCredentialsPrompt: Story = {
  args: {
    messages: [
      {
        type: "message" as const,
        role: "user" as const,
        content: "Run the GitHub agent",
        timestamp: new Date(Date.now() - 3 * 60 * 1000),
      },
      {
        type: "credentials_needed" as const,
        credentials: [
          {
            provider: "github",
            providerName: "GitHub",
            credentialType: "oauth2" as const,
            title: "GitHub Integration",
          },
        ],
        agentName: "GitHub Integration Agent",
        message:
          "To run GitHub Integration Agent, you need to add credentials.",
        timestamp: new Date(Date.now() - 2 * 60 * 1000),
      },
    ],
  },
};

export const WithNoResults: Story = {
  args: {
    messages: [
      {
        type: "message" as const,
        role: "user" as const,
        content: "Find crypto mining agents",
        timestamp: new Date(Date.now() - 3 * 60 * 1000),
      },
      {
        type: "tool_call" as const,
        toolId: "tool-789b1234-c56d-78e9-f012-345678901abc",
        toolName: "find_agent",
        arguments: { query: "crypto mining" },
        timestamp: new Date(Date.now() - 2 * 60 * 1000),
      },
      {
        type: "no_results" as const,
        message:
          "No agents found matching 'crypto mining'. Try different keywords or browse the marketplace.",
        suggestions: [
          "Try more general terms",
          "Browse categories in the marketplace",
          "Check spelling",
        ],
        timestamp: new Date(Date.now() - 1 * 60 * 1000),
      },
    ],
  },
};

export const WithAgentCarousel: Story = {
  args: {
    messages: [
      {
        type: "message" as const,
        role: "user" as const,
        content: "Find automation agents",
        timestamp: new Date(Date.now() - 3 * 60 * 1000),
      },
      {
        type: "tool_call" as const,
        toolId: "tool-321d5678-e90f-12a3-b456-789012345cde",
        toolName: "find_agent",
        arguments: { query: "automation" },
        timestamp: new Date(Date.now() - 2 * 60 * 1000),
      },
      {
        type: "agent_carousel" as const,
        agents: [
          {
            id: "agent-1",
            name: "Email Automation",
            description:
              "Automates email responses based on custom rules and templates",
            version: 1,
          },
          {
            id: "agent-2",
            name: "Social Media Manager",
            description:
              "Schedules and publishes posts across multiple platforms",
            version: 2,
          },
          {
            id: "agent-3",
            name: "Data Sync Agent",
            description: "Syncs data between different services automatically",
            version: 1,
          },
        ],
        totalCount: 15,
        timestamp: new Date(Date.now() - 1 * 60 * 1000),
      },
    ],
  },
};

export const WithExecutionStarted: Story = {
  args: {
    messages: [
      {
        type: "message" as const,
        role: "user" as const,
        content: "Run the data analysis agent",
        timestamp: new Date(Date.now() - 3 * 60 * 1000),
      },
      {
        type: "tool_call" as const,
        toolId: "tool-654f9876-a54b-32c1-d765-432109876fed",
        toolName: "run_agent",
        arguments: { agent_id: "agent-123", input: { file: "data.csv" } },
        timestamp: new Date(Date.now() - 2 * 60 * 1000),
      },
      {
        type: "execution_started" as const,
        executionId: "exec-123e4567-e89b-12d3-a456-426614174000",
        agentName: "Data Analysis Agent",
        message: "Your agent execution has started successfully",
        timestamp: new Date(Date.now() - 1 * 60 * 1000),
      },
    ],
  },
};

export const MixedConversation: Story = {
  args: {
    messages: [
      {
        type: "message" as const,
        role: "user" as const,
        content: "Hello! I want to find and run an automation agent",
        timestamp: new Date(Date.now() - 15 * 60 * 1000),
      },
      {
        type: "message" as const,
        role: "assistant" as const,
        content:
          "I can help you find and run automation agents! Let me search for you.",
        timestamp: new Date(Date.now() - 14 * 60 * 1000),
      },
      {
        type: "tool_call" as const,
        toolId: "tool-111",
        toolName: "find_agent",
        arguments: { query: "automation" },
        timestamp: new Date(Date.now() - 13 * 60 * 1000),
      },
      {
        type: "agent_carousel" as const,
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
        timestamp: new Date(Date.now() - 12 * 60 * 1000),
      },
      {
        type: "message" as const,
        role: "user" as const,
        content: "Run the Email Automation agent",
        timestamp: new Date(Date.now() - 10 * 60 * 1000),
      },
      {
        type: "tool_call" as const,
        toolId: "tool-222",
        toolName: "run_agent",
        arguments: { agent_id: "agent-1" },
        timestamp: new Date(Date.now() - 9 * 60 * 1000),
      },
      {
        type: "credentials_needed" as const,
        credentials: [
          {
            provider: "gmail",
            providerName: "Gmail",
            credentialType: "oauth2" as const,
            title: "Gmail Integration",
          },
        ],
        agentName: "Email Automation",
        message: "To run Email Automation, you need to add credentials.",
        timestamp: new Date(Date.now() - 8 * 60 * 1000),
      },
      {
        type: "message" as const,
        role: "user" as const,
        content: "Try finding crypto agents instead",
        timestamp: new Date(Date.now() - 5 * 60 * 1000),
      },
      {
        type: "tool_call" as const,
        toolId: "tool-333",
        toolName: "find_agent",
        arguments: { query: "crypto" },
        timestamp: new Date(Date.now() - 4 * 60 * 1000),
      },
      {
        type: "no_results" as const,
        message: "No agents found matching 'crypto'. Try different keywords.",
        suggestions: ["Try more general terms", "Browse the marketplace"],
        timestamp: new Date(Date.now() - 3 * 60 * 1000),
      },
    ],
  },
};
