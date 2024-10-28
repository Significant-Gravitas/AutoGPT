import type { Meta, StoryObj } from "@storybook/react";
import { FeaturedStoreCard } from "./FeaturedStoreCard";
import { userEvent, within } from "@storybook/test";

const meta = {
  title: "AGPT UI/Featured Store Card",
  component: FeaturedStoreCard,
  parameters: {
    layout: {
      center: true,
      padding: 0,
    },
  },
  tags: ["autodocs"],
  argTypes: {
    agentName: { control: "text" },
    subHeading: { control: "text" },
    agentImage: { control: "text" },
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
    subHeading: "Optimize your website's SEO",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
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
    subHeading: "Basic data analysis tool",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
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
    subHeading: "Your AI coding companion",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
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
    subHeading: "All-in-one productivity suite",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorName: "Productivity Plus",
    description:
      "An all-in-one productivity suite that helps you manage tasks, schedule meetings, track time, and collaborate with team members. Powered by advanced AI to optimize your workflow and boost efficiency.",
    runs: 75000,
    rating: 4.5,
    onClick: () => console.log("Card clicked"),
  },
};

export const ShortDescription: Story = {
  args: {
    agentName: "QuickTask",
    subHeading: "Fast task automation",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorName: "EfficientWorks",
    description: "Streamline your workflow.",
    runs: 50000,
    rating: 4.2,
    onClick: () => console.log("Card clicked"),
  },
};

export const TwoLineName: Story = {
  args: {
    agentName: "Agent name goes here example of agent with two lines of text",
    subHeading: "Multi-line agent example",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorName: "InnovativeTech Solutions",
    description:
      "Boost your productivity with our cutting-edge AI assistant. Manages tasks, schedules, and more.",
    runs: 100000,
    rating: 4.8,
    onClick: () => console.log("Card clicked"),
  },
};

export const TwoLineNameLongDescription: Story = {
  args: {
    agentName: "Advanced Natural Language Processing and Machine Learning",
    subHeading: "State-of-the-art NLP & ML",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorName: "AI Research Labs",
    description:
      "Our cutting-edge AI assistant combines state-of-the-art natural language processing and machine learning algorithms to provide unparalleled support in various domains. From text analysis and sentiment detection to predictive modeling and data visualization, this powerful tool empowers researchers, data scientists, and businesses to unlock valuable insights from complex datasets and textual information. With continuous learning capabilities and adaptable interfaces, it's the perfect companion for pushing the boundaries of AI-driven discovery and innovation.",
    runs: 150000,
    rating: 4.9,
    onClick: () => console.log("Card clicked"),
  },
};

export const WithInteraction: Story = {
  args: {
    agentName: "AI Writing Assistant",
    subHeading: "Enhance your writing",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
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
