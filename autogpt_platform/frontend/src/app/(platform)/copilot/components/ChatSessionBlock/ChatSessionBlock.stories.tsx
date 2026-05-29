import type { ComponentProps } from "react";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { ChatSessionBlock } from "./ChatSessionBlock";

function daysAgo(days: number) {
  const date = new Date();
  date.setDate(date.getDate() - days);
  return date.toISOString();
}

function renderSessionBlock(args: ComponentProps<typeof ChatSessionBlock>) {
  return (
    <button
      type="button"
      className={[
        "w-full rounded-lg px-3 py-2.5 text-left transition-colors",
        args.isActive ? "bg-zinc-100" : "bg-white hover:bg-zinc-50",
      ].join(" ")}
    >
      <ChatSessionBlock {...args} />
    </button>
  );
}

const meta: Meta<typeof ChatSessionBlock> = {
  title: "Copilot/ChatSessionBlock",
  component: ChatSessionBlock,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Copilot session row content used in the chat sidebar and mobile drawer, including external-platform origin logos.",
      },
    },
  },
  decorators: [
    (Story) => (
      <div className="w-80 rounded-lg border border-zinc-100 bg-zinc-50 p-3">
        <Story />
      </div>
    ),
  ],
  render: renderSessionBlock,
  args: {
    title: "Review Discord support flow",
    updatedAt: new Date().toISOString(),
    sourcePlatform: "discord",
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Discord: Story = {};

export const SlackActive: Story = {
  args: {
    title: "Triage Slack escalation",
    updatedAt: daysAgo(1),
    sourcePlatform: "slack",
    isActive: true,
  },
};

export const TelegramQueued: Story = {
  args: {
    title: "Telegram onboarding message",
    updatedAt: daysAgo(2),
    sourcePlatform: "telegram",
    chatStatus: "queued",
  },
};

export const GitHubRunning: Story = {
  args: {
    title: "Investigate GitHub issue",
    updatedAt: daysAgo(3),
    sourcePlatform: "github",
    chatStatus: "running",
  },
};

export const LinearCompleted: Story = {
  args: {
    title: "Summarize Linear ticket",
    updatedAt: daysAgo(4),
    sourcePlatform: "linear",
    showCompleted: true,
  },
};

export const PlatformMatrix: Story = {
  render: () => (
    <div className="space-y-1">
      {[
        {
          title: "Discord community report",
          sourcePlatform: "discord",
          updatedAt: new Date().toISOString(),
        },
        {
          title: "Slack customer question",
          sourcePlatform: "slack",
          updatedAt: daysAgo(1),
          isActive: true,
        },
        {
          title: "Telegram lead follow-up",
          sourcePlatform: "telegram",
          updatedAt: daysAgo(2),
          chatStatus: "queued",
        },
        {
          title: "GitHub issue analysis",
          sourcePlatform: "github",
          updatedAt: daysAgo(3),
          chatStatus: "running",
        },
        {
          title: "Linear sprint update",
          sourcePlatform: "linear",
          updatedAt: daysAgo(4),
          showCompleted: true,
        },
        {
          title: "Native Copilot chat",
          updatedAt: daysAgo(8),
        },
      ].map((session) => (
        <div key={session.title}>{renderSessionBlock(session)}</div>
      ))}
    </div>
  ),
};
