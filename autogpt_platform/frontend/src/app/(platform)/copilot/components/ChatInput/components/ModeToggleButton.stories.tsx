import type { Meta, StoryObj } from "@storybook/nextjs";
import { ModeToggleButton } from "./ModeToggleButton";

const meta: Meta<typeof ModeToggleButton> = {
  title: "Copilot/ModeToggleButton",
  component: ModeToggleButton,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Toggle between Fast and Extended Thinking copilot modes. Disabled while a response is streaming.",
      },
    },
  },
  args: {
    onToggle: () => {},
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const FastMode: Story = {
  args: {
    mode: "fast",
    isStreaming: false,
  },
};

export const ExtendedThinkingMode: Story = {
  args: {
    mode: "extended_thinking",
    isStreaming: false,
  },
};

export const DisabledWhileStreaming: Story = {
  args: {
    mode: "fast",
    isStreaming: true,
  },
};
