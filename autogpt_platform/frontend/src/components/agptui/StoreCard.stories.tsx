import type { Meta, StoryObj } from "@storybook/react";
import { StoreCard } from "./StoreCard";
import { userEvent, within, expect } from "@storybook/test";

const meta = {
  title: "new/StoreCard",
  component: StoreCard,

  decorators: [
    (Story) => (
      <div className="flex items-center justify-center p-4">
        <Story />
      </div>
    ),
  ],
  tags: ["autodocs"],
  argTypes: {
    agentName: { control: "text" },
    agentImage: { control: "text" },
    description: { control: "text" },
    runs: { control: "number" },
    rating: { control: "number", min: 0, max: 5, step: 0.1 },
    onClick: { action: "clicked" },
    avatarSrc: { control: "text" },
    hideAvatar: { control: "boolean" },
    creatorName: { control: "text" },
  },
} satisfies Meta<typeof StoreCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    agentName: "SEO Optimizer",
    agentImage: "default_agent_image.jpg",
    description:
      "Optimize your website's SEO with AI-powered suggestions and best practices. Get detailed reports and actionable recommendations.",
    runs: 10000,
    rating: 4.5,
    onClick: () => console.log("Default StoreCard clicked"),
    avatarSrc: "default_avatar.png",
    creatorName: "AI Solutions Inc.",
  },
};

export const LowRating: Story = {
  args: {
    agentName: "Data Analyzer",
    agentImage: "default_agent_image.jpg",
    description:
      "Analyze complex datasets with machine learning algorithms and statistical models for insights.",
    runs: 5000,
    rating: 2.7,
    onClick: () => console.log("LowRating StoreCard clicked"),
    avatarSrc: "default_avatar.png",
    creatorName: "DataTech",
  },
};

export const HighRuns: Story = {
  args: {
    agentName: "Code Assistant",
    agentImage: "default_agent_image.jpg",
    description:
      "Get AI-powered coding help for various programming languages. Debug issues, optimize code, and learn new patterns.",
    runs: 1000000,
    rating: 4.8,
    onClick: () => console.log("HighRuns StoreCard clicked"),
    avatarSrc: "default_avatar.png",
    creatorName: "DevTools Co.",
  },
};

export const WithInteraction: Story = {
  args: {
    agentName: "Task Planner",
    agentImage: "default_agent_image.jpg",
    description:
      "Plan and organize your tasks efficiently with AI assistance. Set priorities, deadlines, and track progress.",
    runs: 50000,
    rating: 4.2,
    onClick: () => console.log("WithInteraction StoreCard clicked"),
    avatarSrc: "default_avatar.png",
    creatorName: "Productivity Plus",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const storeCard = canvas.getByTestId("store-card");

    await userEvent.hover(storeCard);
    await new Promise((resolve) => setTimeout(resolve, 300));
    await userEvent.click(storeCard);
  },
};

export const LongDescription: Story = {
  args: {
    agentName: "AI Writing Assistant",
    agentImage: "default_agent_image.jpg",
    description:
      "Enhance your writing with our advanced AI-powered assistant. It offers real-time suggestions for grammar, style, and tone, helps with research and fact-checking, and provides vocabulary enhancements for more engaging content. Perfect for content creators, marketers, and writers of all levels.",
    runs: 75000,
    rating: 4.7,
    onClick: () => console.log("LongDescription StoreCard clicked"),
    avatarSrc: "default_avatar.png",
    creatorName: "ContentGenius",
  },
};

export const HiddenAvatar: Story = {
  args: {
    agentName: "Data Visualizer",
    agentImage: "default_agent_image.jpg",
    description:
      "Create stunning visualizations from complex datasets. Generate charts, graphs, and interactive dashboards.",
    runs: 60000,
    rating: 4.6,
    onClick: () => console.log("HiddenAvatar StoreCard clicked"),
    avatarSrc: "default_avatar.png",
    hideAvatar: true,
  },
};

export const LongTitle: Story = {
  args: {
    agentName:
      "Universal Language Translator Pro with Advanced Neural Network Technology",
    agentImage: "default_agent_image.jpg",
    description:
      "Breaking language barriers with cutting-edge AI translation technology for global communication.",
    runs: 120000,
    rating: 4.9,
    onClick: () => console.log("LongTitle StoreCard clicked"),
    avatarSrc: "default_avatar.png",
    creatorName: "Global Linguistics Technologies",
  },
};

export const ZeroValues: Story = {
  args: {
    agentName: "New Project Template",
    agentImage: "default_agent_image.jpg",
    description:
      "A basic template for new projects with no user data or ratings yet.",
    runs: 0,
    rating: 0,
    onClick: () => console.log("ZeroValues StoreCard clicked"),
    avatarSrc: "default_avatar.png",
    creatorName: "Template Systems",
  },
};

export const MobileView: Story = {
  args: {
    agentName: "SEO Optimizer",
    agentImage: "default_agent_image.jpg",
    description:
      "Optimize your website's SEO with AI-powered suggestions and best practices. Get detailed reports and actionable recommendations.",
    runs: 10000,
    rating: 4.5,
    onClick: () => console.log("MobileView StoreCard clicked"),
    avatarSrc: "default_avatar.png",
    creatorName: "AI Solutions Inc.",
  },
  parameters: {
    viewport: {
      defaultViewport: "mobile2",
    },
  },
};
