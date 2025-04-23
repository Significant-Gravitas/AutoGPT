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
    name: { control: "text" },
    creator: { control: "text" },
    shortDescription: { control: "text" },
    longDescription: { control: "text" },
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
    name: "AI Video Generator",
    storeListingVersionId: "123",
    creator: "Toran Richards",
    shortDescription:
      "Transform ideas into breathtaking images with this AI-powered Image Generator.",
    longDescription: `Create Viral-Ready Content in Seconds! Transform trending topics into engaging videos with this cutting-edge AI Video Generator. Perfect for content creators, social media managers, and marketers looking to quickly produce high-quality content.

Key features include:
- Customizable video output
- 15+ pre-made templates
- Auto scene detection
- Smart text-to-speech
- Multiple export formats
- SEO-optimized suggestions`,
    rating: 4.7,
    runs: 1500,
    categories: ["Video", "Content Creation", "Social Media"],
    lastUpdated: "2 days ago",
    version: "1.2.0",
  },
};

export const LowRating: Story = {
  args: {
    ...Default.args,
    name: "Data Analyzer",
    creator: "DataTech",
    shortDescription:
      "Analyze complex datasets with machine learning algorithms",
    longDescription:
      "A comprehensive data analysis tool that leverages machine learning to provide deep insights into your datasets. Currently in beta testing phase.",
    rating: 2.7,
    runs: 5000,
    categories: ["Data Analysis", "Machine Learning"],
    lastUpdated: "1 week ago",
    version: "0.9.5",
  },
};

export const HighRuns: Story = {
  args: {
    ...Default.args,
    name: "Code Assistant",
    creator: "DevAI",
    shortDescription:
      "Get AI-powered coding help for various programming languages",
    longDescription:
      "An advanced AI coding assistant that supports multiple programming languages and frameworks. Features include code completion, refactoring suggestions, and bug detection.",
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
    shortDescription: "Plan and organize your tasks efficiently with AI",
    longDescription:
      "An intelligent task management system that helps you organize, prioritize, and complete your tasks more efficiently. Features smart scheduling and AI-powered suggestions.",
    rating: 4.2,
    runs: 50000,
    categories: ["Productivity", "Task Management", "AI"],
    lastUpdated: "3 days ago",
    version: "1.5.2",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);

    // Test run agent button
    const runButton = canvas.getByText("Run agent");
    await userEvent.hover(runButton);
    await userEvent.click(runButton);

    // Test rating interaction
    const ratingStars = canvas.getAllByLabelText(/Star Icon/);
    await userEvent.hover(ratingStars[3]);
    await userEvent.click(ratingStars[3]);

    // Test category interaction
    const category = canvas.getByText("Productivity");
    await userEvent.hover(category);
    await userEvent.click(category);
  },
};

export const LongDescription: Story = {
  args: {
    ...Default.args,
    name: "AI Writing Assistant",
    creator: "WordCraft AI",
    shortDescription:
      "Enhance your writing with our advanced AI-powered assistant.",
    longDescription:
      "It offers real-time suggestions for grammar, style, and tone, helps with research and fact-checking, and can even generate content ideas based on your input.",
    rating: 4.7,
    runs: 75000,
    categories: ["Writing", "AI", "Content Creation"],
    lastUpdated: "5 days ago",
    version: "3.0.1",
  },
};
