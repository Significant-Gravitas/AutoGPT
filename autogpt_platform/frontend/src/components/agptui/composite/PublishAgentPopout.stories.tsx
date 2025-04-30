import type { Meta, StoryObj } from "@storybook/react";
import { PublishAgentPopout } from "@/components/agptui/composite/PublishAgentPopout";
import { userEvent, within, expect } from "@storybook/test";

const meta = {
  title: "AGPT UI/Composite/Publish Agent Popout",
  component: PublishAgentPopout,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    trigger: { control: "object" },
  },
} satisfies Meta<typeof PublishAgentPopout>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {},
};

export const WithCustomTrigger: Story = {
  args: {
    trigger: <button>Custom Publish Button</button>,
  },
};
