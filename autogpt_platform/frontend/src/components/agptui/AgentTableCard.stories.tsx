import type { Meta, StoryObj } from "@storybook/react";
import { AgentTableCard } from "./AgentTableCard";
import { userEvent, within, expect } from "@storybook/test";
import { type StatusType } from "./Status";

const meta: Meta<typeof AgentTableCard> = {
  title: "AGPT UI/Agent Table Card",
  component: AgentTableCard,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof AgentTableCard>;

export const Default: Story = {
  args: {
    agentName: "Super Coder",
    description: "An AI agent that writes clean, efficient code",
    imageSrc: [
      "https://ddz4ak4pa3d19.cloudfront.net/cache/53/b2/53b2bc7d7900f0e1e60bf64ebf38032d.jpg",
    ],
    dateSubmitted: "2023-05-15",
    status: "ACTIVE" as StatusType,
    runs: 1500,
    rating: 4.8,
  },
};

export const NoRating: Story = {
  args: {
    ...Default.args,
    rating: undefined,
  },
};

export const NoRuns: Story = {
  args: {
    ...Default.args,
    runs: undefined,
  },
};

export const InactiveAgent: Story = {
  args: {
    ...Default.args,
    status: "INACTIVE" as StatusType,
  },
};

export const LongDescription: Story = {
  args: {
    ...Default.args,
    description:
      "This is a very long description that should wrap to multiple lines. It contains detailed information about the agent and its capabilities.",
  },
};

export const InteractionTest: Story = {
  ...Default,
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const moreButton = canvas.getByRole("button");
    await userEvent.click(moreButton);
  },
};
