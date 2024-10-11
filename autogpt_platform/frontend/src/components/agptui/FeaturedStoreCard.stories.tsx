import type { Meta, StoryObj } from "@storybook/react";
import { FeaturedStoreCard } from "./FeaturedStoreCard";
import { userEvent, within } from "@storybook/test";

const meta = {
  title: "AGPTUI/FeaturedStoreCard",
  component: FeaturedStoreCard,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    agentName: { control: "text" },
    creatorName: { control: "text" },
    description: { control: "text" },
    runs: { control: "number" },
    rating: { control: "number", min: 0, max: 5, step: 0.1 },
    onClick: { action: "clicked" },
  },
} satisfies Meta<typeof FeaturedStoreCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    agentName: "SEO Optimizer Pro",
    creatorName: "AI Solutions Inc.",
    description:
      "Boost your website's search engine rankings with our advanced AI-powered SEO optimization tool.",
    runs: 50000,
    rating: 4.7,
    onClick: () => console.log("Card clicked"),
  },
};

export const LowRating: Story = {
  args: {
    agentName: "Data Analyzer Lite",
    creatorName: "DataTech",
    description: "A basic tool for analyzing small to medium-sized datasets.",
    runs: 10000,
    rating: 2.8,
    onClick: () => console.log("Card clicked"),
  },
};

export const HighRuns: Story = {
  args: {
    agentName: "CodeAssist AI",
    creatorName: "DevTools Co.",
    description:
      "Get instant coding help and suggestions for multiple programming languages.",
    runs: 1000000,
    rating: 4.9,
    onClick: () => console.log("Card clicked"),
  },
};

export const LongDescription: Story = {
  args: {
    agentName: "MultiTasker",
    creatorName: "Productivity Plus",
    description:
      "An all-in-one productivity suite that helps you manage tasks, schedule meetings, track time, and collaborate with team members. Powered by advanced AI to optimize your workflow and boost efficiency.",
    runs: 75000,
    rating: 4.5,
    onClick: () => console.log("Card clicked"),
  },
};

export const WithInteraction: Story = {
  args: {
    agentName: "AI Writing Assistant",
    creatorName: "WordCraft AI",
    description:
      "Enhance your writing with AI-powered suggestions, grammar checks, and style improvements.",
    runs: 200000,
    rating: 4.6,
    onClick: () => console.log("Card clicked"),
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const featuredCard = canvas.getByText("AI Writing Assistant");

    await userEvent.hover(featuredCard);
    await userEvent.click(featuredCard);
  },
};
