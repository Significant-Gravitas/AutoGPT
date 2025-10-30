import type { Meta, StoryObj } from "@storybook/nextjs";
import { ChatMessage } from "./ChatMessage";

const meta = {
  title: "Molecules/ChatMessage",
  component: ChatMessage,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof ChatMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const UserMessage: Story = {
  args: {
    message: {
      type: "message",
      role: "user",
      content: "Hello! How can you help me today?",
      timestamp: new Date(Date.now() - 2 * 60 * 1000), // 2 minutes ago
    },
  },
};

export const AssistantMessage: Story = {
  args: {
    message: {
      type: "message",
      role: "assistant",
      content:
        "I can help you discover and run AI agents! I can search for agents, explain what they do, help you set them up, and run them for you. What would you like to do?",
      timestamp: new Date(Date.now() - 1 * 60 * 1000), // 1 minute ago
    },
  },
};

export const UserMessageLong: Story = {
  args: {
    message: {
      type: "message",
      role: "user",
      content:
        "I'm looking for an agent that can help me analyze data from a CSV file. I have sales data for the last quarter and I want to understand trends, identify top-performing products, and get recommendations for inventory management. Can you help me find something suitable?",
      timestamp: new Date(Date.now() - 5 * 60 * 1000), // 5 minutes ago
    },
  },
};

export const AssistantMessageLong: Story = {
  args: {
    message: {
      type: "message",
      role: "assistant",
      content:
        "Great! I found several agents that can help with data analysis. The 'CSV Data Analyzer' agent is perfect for your needs. It can:\n\n1. Parse and validate CSV files\n2. Generate statistical summaries\n3. Identify trends and patterns\n4. Create visualizations\n5. Provide actionable insights\n\nWould you like me to set this up for you?",
      timestamp: new Date(Date.now() - 30 * 1000), // 30 seconds ago
    },
  },
};

export const JustNow: Story = {
  args: {
    message: {
      type: "message",
      role: "user",
      content: "Yes, please set it up!",
    },
  },
};

export const Conversation: Story = {
  args: {
    message: {
      type: "message",
      role: "user",
      content: "",
    },
  },
  render: () => (
    <div className="w-full max-w-2xl space-y-0">
      <ChatMessage
        message={{
          type: "message",
          role: "user",
          content: "Find me automation agents",
          timestamp: new Date(Date.now() - 10 * 60 * 1000),
        }}
      />
      <ChatMessage
        message={{
          type: "message",
          role: "assistant",
          content:
            "I found 15 automation agents. Here are the top 3:\n\n1. Email Automation Agent\n2. Social Media Scheduler\n3. Workflow Automator\n\nWould you like details on any of these?",
          timestamp: new Date(Date.now() - 9 * 60 * 1000),
        }}
      />
      <ChatMessage
        message={{
          type: "message",
          role: "user",
          content: "Tell me about the Email Automation Agent",
          timestamp: new Date(Date.now() - 8 * 60 * 1000),
        }}
      />
      <ChatMessage
        message={{
          type: "message",
          role: "assistant",
          content:
            "The Email Automation Agent helps you automate email workflows. It can send scheduled emails, respond to incoming messages based on rules, and integrate with your calendar for meeting reminders.",
          timestamp: new Date(Date.now() - 7 * 60 * 1000),
        }}
      />
    </div>
  ),
};
