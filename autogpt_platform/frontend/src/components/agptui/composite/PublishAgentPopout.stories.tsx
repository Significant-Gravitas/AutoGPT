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

export const PublishFlow: Story = {
  args: {},
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);

    // Open popout
    const publishButton = canvas.getByText("Publish Agent");
    await userEvent.click(publishButton);

    // Select an agent (assuming one exists)
    const agentCard = await canvas.findByRole("button", {
      name: /select agent/i,
    });
    await userEvent.click(agentCard);

    // Click next
    const nextButton = canvas.getByText("Next");
    await userEvent.click(nextButton);

    // Fill out info form
    // Note: Actual form interactions would need to be added based on PublishAgentInfo implementation
  },
};
