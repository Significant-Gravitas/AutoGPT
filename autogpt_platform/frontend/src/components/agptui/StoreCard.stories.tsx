import type { Meta, StoryObj } from "@storybook/react";
import { StoreCard } from "./StoreCard";
import { userEvent, within } from "@storybook/test";

const meta = {
  title: "AGPTUI/StoreCard",
  component: StoreCard,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    agentName: { control: "text" },
    description: { control: "text" },
    runs: { control: "number" },
    rating: { control: "number", min: 0, max: 5, step: 0.1 },
    onClick: { action: "clicked" },
  },
} satisfies Meta<typeof StoreCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    agentName: "SEO Optimizer",
    description: "Optimize your website's SEO with AI-powered suggestions",
    runs: 10000,
    rating: 4.5,
    onClick: () => console.log("Default StoreCard clicked"),
  },
};

export const LowRating: Story = {
  args: {
    agentName: "Data Analyzer",
    description: "Analyze complex datasets with machine learning algorithms",
    runs: 5000,
    rating: 2.7,
    onClick: () => console.log("LowRating StoreCard clicked"),
  },
};

export const HighRuns: Story = {
  args: {
    agentName: "Code Assistant",
    description: "Get AI-powered coding help for various programming languages",
    runs: 1000000,
    rating: 4.8,
    onClick: () => console.log("HighRuns StoreCard clicked"),
  },
};

export const WithInteraction: Story = {
  args: {
    agentName: "Task Planner",
    description: "Plan and organize your tasks efficiently with AI",
    runs: 50000,
    rating: 4.2,
    onClick: () => console.log("WithInteraction StoreCard clicked"),
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const storeCard = canvas.getByText("Task Planner");

    await userEvent.hover(storeCard);
    await userEvent.click(storeCard);
  },
};
