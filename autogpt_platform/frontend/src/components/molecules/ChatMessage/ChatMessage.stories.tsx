import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { AgentStatusAvatar } from "../AgentStatusAvatar/AgentStatusAvatar";
import { AUTOPILOT_AVATAR_URL } from "../AutopilotAvatar/helpers";
import { ChatMessage } from "./ChatMessage";

const meta = {
  title: "Molecules/ChatMessage",
  component: ChatMessage,
  tags: ["autodocs"],
  parameters: { layout: "padded", a11y: { test: "error" } },
  args: {
    from: "user",
    children: "Can you find me three hotels in Lisbon for next weekend?",
  },
} satisfies Meta<typeof ChatMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const User: Story = {};

export const Agent: Story = {
  args: {
    from: "agent",
    children:
      "Here are three places with good reviews and rooms free on Saturday.",
  },
};

export const AgentWithAvatar: Story = {
  args: {
    from: "agent",
    avatar: (
      <AgentStatusAvatar
        name="Otto"
        src={AUTOPILOT_AVATAR_URL}
        status="done"
        size="sm"
      />
    ),
    children: "Done. I booked the second one.",
  },
};

export const Conversation: Story = {
  render: () => (
    <div className="flex max-w-md flex-col gap-4">
      <ChatMessage from="user">What is on my calendar tomorrow?</ChatMessage>
      <ChatMessage from="agent">
        Two meetings: a design review at 10 and lunch with Sam at 12:30.
      </ChatMessage>
      <ChatMessage from="user">Move the review to Thursday.</ChatMessage>
    </div>
  ),
};
