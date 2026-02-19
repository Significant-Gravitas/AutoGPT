import type { Meta, StoryObj } from "@storybook/nextjs";
import { fn } from "@storybook/test";
import { ChatInput } from "./ChatInput";

const meta: Meta<typeof ChatInput> = {
  title: "CoPilot/Chat/ChatInput",
  component: ChatInput,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Chat message input with send button, voice recording, and streaming stop support.",
      },
    },
  },
  decorators: [
    (Story) => (
      <div className="max-w-[600px]">
        <Story />
      </div>
    ),
  ],
  args: {
    onSend: fn(),
  },
};
export default meta;
type Story = StoryObj<typeof ChatInput>;

export const Default: Story = {};

export const Disabled: Story = {
  args: { disabled: true },
};

export const Streaming: Story = {
  args: {
    isStreaming: true,
    onStop: fn(),
  },
};

export const WithPlaceholder: Story = {
  args: { placeholder: "Ask me anything about your agents..." },
};
