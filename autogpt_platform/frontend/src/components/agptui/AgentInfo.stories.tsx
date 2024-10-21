import type { Meta, StoryObj } from "@storybook/react";
import { AgentInfo } from "./AgentInfo";
import { userEvent, within } from "@storybook/test";

const meta = {
  title: "AGPT UI/Agent Info",
  component: AgentInfo,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    onRunAgent: { action: "run agent clicked" },
    name: { control: "text" },
    creator: { control: "text" },
    description: { control: "text" },
    rating: { control: "number", min: 0, max: 5, step: 0.1 },
    runs: { control: "number" },
    categories: { control: "object" },
    lastUpdated: { control: "text" },
    version: { control: "text" },
  },
} satisfies Meta<typeof AgentInfo>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    onRunAgent: () => console.log("Run agent clicked"),
    name: "SEO Optimizer",
    creator: "AI Labs",
    description: "Optimize your website's SEO with AI-powered suggestions",
    rating: 4.5,
    runs: 10000,
    categories: ["SEO", "Marketing", "AI"],
    lastUpdated: "2 days ago",
    version: "1.2.0",
  },
};

export const LowRating: Story = {
  args: {
    ...Default.args,
    name: "Data Analyzer",
    creator: "DataTech",
    description: "Analyze complex datasets with machine learning algorithms",
    rating: 2.7,
    runs: 5000,
    categories: ["Data Analysis"],
    lastUpdated: "1 week ago",
    version: "0.9.5",
  },
};

export const HighRuns: Story = {
  args: {
    ...Default.args,
    name: "Code Assistant",
    creator: "DevAI",
    description: "Get AI-powered coding help for various programming languages",
    rating: 4.8,
    runs: 1000000,
    categories: ["Programming", "AI", "Developer Tools"],
    lastUpdated: "1 day ago",
    version: "2.1.3",
  },
};

export const WithInteraction: Story = {
  args: {
    ...Default.args,
    name: "Task Planner",
    creator: "Productivity AI",
    description: "Plan and organize your tasks efficiently with AI",
    rating: 4.2,
    runs: 50000,
    categories: ["Productivity", "Task Management", "AI"],
    lastUpdated: "3 days ago",
    version: "1.5.2",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const runButton = canvas.getByText("Run agent");

    await userEvent.hover(runButton);
    await userEvent.click(runButton);
  },
};

export const LongDescription: Story = {
  args: {
    ...Default.args,
    name: "AI Writing Assistant",
    creator: "WordCraft AI",
    description:
      "Enhance your writing with our advanced AI-powered assistant. It offers real-time suggestions for grammar, style, and tone, helps with research and fact-checking, and can even generate content ideas based on your input.",
    rating: 4.7,
    runs: 75000,
    categories: ["Writing", "AI", "Content Creation"],
    lastUpdated: "5 days ago",
    version: "3.0.1",
  },
};
