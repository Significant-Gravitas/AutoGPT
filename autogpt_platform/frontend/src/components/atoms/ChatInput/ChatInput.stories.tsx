import type { Meta, StoryObj } from "@storybook/nextjs";
import { ChatInput } from "./ChatInput";

const meta = {
  title: "Atoms/ChatInput",
  component: ChatInput,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
  args: {
    onSend: (message: string) => console.log("Message sent:", message),
  },
} satisfies Meta<typeof ChatInput>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    placeholder: "Type your message...",
    disabled: false,
  },
};

export const Disabled: Story = {
  args: {
    placeholder: "Type your message...",
    disabled: true,
  },
};

export const CustomPlaceholder: Story = {
  args: {
    placeholder: "Ask me anything about agents...",
    disabled: false,
  },
};

export const WithText: Story = {
  render: (args) => {
    return (
      <div className="space-y-4">
        <ChatInput {...args} />
        <p className="text-sm text-neutral-600 dark:text-neutral-400">
          Try typing a message and pressing Enter to send, or Shift+Enter for a
          new line.
        </p>
      </div>
    );
  },
  args: {
    placeholder: "Type your message...",
  },
};
