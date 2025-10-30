import type { Meta, StoryObj } from "@storybook/nextjs";
import { MessageBubble } from "./MessageBubble";

const meta = {
  title: "Atoms/MessageBubble",
  component: MessageBubble,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof MessageBubble>;

export default meta;
type Story = StoryObj<typeof meta>;

export const User: Story = {
  args: {
    variant: "user",
    children: "Hello! This is a message from the user.",
  },
};

export const Assistant: Story = {
  args: {
    variant: "assistant",
    children:
      "Hi there! This is a response from the AI assistant. It can be longer and contain multiple sentences to show how the bubble handles different content lengths.",
  },
};

export const UserLong: Story = {
  args: {
    variant: "user",
    children:
      "This is a much longer message from the user that demonstrates how the message bubble handles multi-line content. It should wrap nicely and maintain good readability even with lots of text. The styling should remain consistent regardless of the content length.",
  },
};

export const AssistantWithCode: Story = {
  args: {
    variant: "assistant",
    children: (
      <div>
        <p className="mb-2">Here&apos;s a code example:</p>
        <code className="block rounded bg-neutral-100 p-2 dark:bg-neutral-800">
          const greeting = &quot;Hello, world!&quot;;
        </code>
      </div>
    ),
  },
};
