import type { Meta, StoryObj } from "@storybook/react";
import { StoreCard } from "./StoreCard";
import { userEvent, within, expect } from "@storybook/test";

const meta = {
  title: "AGPT UI/StoreCard",
  component: StoreCard,
  parameters: {
    layout: {
      center: true,
      fullscreen: true,
      padding: 0,
    },
  },
  tags: ["autodocs"],
  argTypes: {
    agentName: { control: "text" },
    agentImage: { control: "text" },
    description: { control: "text" },
    runs: { control: "number" },
    rating: { control: "number", min: 0, max: 5, step: 0.1 },
    onClick: { action: "clicked" },
    avatarSrc: { control: "text" },
  },
} satisfies Meta<typeof StoreCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    agentName: "SEO Optimizer",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    description: "Optimize your website's SEO with AI-powered suggestions",
    runs: 10000,
    rating: 4.5,
    onClick: () => console.log("Default StoreCard clicked"),
    avatarSrc: "https://github.com/shadcn.png",
  },
};

export const LowRating: Story = {
  args: {
    agentName: "Data Analyzer",
    agentImage:
      "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
    description: "Analyze complex datasets with machine learning algorithms",
    runs: 5000,
    rating: 2.7,
    onClick: () => console.log("LowRating StoreCard clicked"),
    avatarSrc: "https://example.com/avatar2.jpg",
  },
};

export const HighRuns: Story = {
  args: {
    agentName: "Code Assistant",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    description: "Get AI-powered coding help for various programming languages",
    runs: 1000000,
    rating: 4.8,
    onClick: () => console.log("HighRuns StoreCard clicked"),
    avatarSrc: "https://example.com/avatar3.jpg",
  },
};

export const WithInteraction: Story = {
  args: {
    agentName: "Task Planner",
    agentImage:
      "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
    description: "Plan and organize your tasks efficiently with AI",
    runs: 50000,
    rating: 4.2,
    onClick: () => console.log("WithInteraction StoreCard clicked"),
    avatarSrc: "https://example.com/avatar4.jpg",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const storeCard = canvas.getByText("Task Planner");

    await userEvent.hover(storeCard);
    await userEvent.click(storeCard);
  },
};

export const LongDescription: Story = {
  args: {
    agentName: "AI Writing Assistant",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    description:
      "Enhance your writing with our advanced AI-powered assistant. It offers real-time suggestions for grammar, style, and tone, helps with research and fact-checking.",
    runs: 75000,
    rating: 4.7,
    onClick: () => console.log("LongDescription StoreCard clicked"),
    avatarSrc: "https://example.com/avatar5.jpg",
  },
};

export const HiddenAvatar: Story = {
  args: {
    agentName: "Data Visualizer",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    description: "Create stunning visualizations from complex datasets",
    runs: 60000,
    rating: 4.6,
    onClick: () => console.log("HiddenAvatar StoreCard clicked"),
    avatarSrc: "https://example.com/avatar6.jpg",
    hideAvatar: true,
  },
};
