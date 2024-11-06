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
    creatorImage: { control: "text" },
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
    agentName: "Personalized Morning Coffee Newsletter example of three lines",
    subHeading:
      "Transform ideas into breathtaking images with this AI-powered Image Generator.",
    description:
      "Elevate your web content with this powerful AI Webpage Copy Improver. Designed for marketers, SEO specialists, and web developers, this tool analyses and enhances website copy for maximum impact. Using advanced language models, it optimizes text for better clarity, SEO performance, and increased conversion rates. The AI examines your existing content, identifies areas for improvement, and generates refined copy that maintains your brand voice while boosting engagement. From homepage headlines to product descriptions, transform your web presence with AI-driven insights. Improve readability, incorporate targeted keywords, and craft compelling calls-to-action - all with the click of a button. Take your digital marketing to the next level with the AI Webpage Copy Improver.",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorName: "AI Solutions Inc.",
    runs: 50000,
    rating: 4.7,
    onClick: () => console.log("Card clicked"),
  },
};

export const LowRating: Story = {
  args: {
    agentName: "Data Analyzer Lite",
    subHeading: "Basic data analysis tool",
    description:
      "A lightweight data analysis tool for basic data processing needs.",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorName: "DataTech",
    runs: 10000,
    rating: 2.8,
    onClick: () => console.log("Card clicked"),
  },
};

export const HighRuns: Story = {
  args: {
    agentName: "CodeAssist AI",
    subHeading: "Your AI coding companion",
    description:
      "An intelligent coding assistant that helps developers write better code faster.",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorName: "DevTools Co.",
    runs: 1000000,
    rating: 4.9,
    onClick: () => console.log("Card clicked"),
  },
};

export const LongDescription: Story = {
  args: {
    agentName: "MultiTasker",
    subHeading: "All-in-one productivity suite",
    description:
      "A comprehensive productivity suite that combines task management, note-taking, and project planning into one seamless interface. Features include smart task prioritization, automated scheduling, and AI-powered insights to help you work more efficiently.",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorName: "Productivity Plus",
    runs: 75000,
    rating: 4.5,
    onClick: () => console.log("Card clicked"),
  },
};

export const ShortDescription: Story = {
  args: {
    agentName: "QuickTask",
    subHeading: "Fast task automation",
    description: "Simple and efficient task automation tool.",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorName: "EfficientWorks",
    runs: 50000,
    rating: 4.2,
    onClick: () => console.log("Card clicked"),
  },
};

export const TwoLineName: Story = {
  args: {
    agentName: "Agent name goes here example of agent with two lines of text",
    subHeading: "Multi-line agent example",
    description:
      "An example agent showcasing how the card handles multi-line agent names.",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorName: "InnovativeTech Solutions",
    runs: 100000,
    rating: 4.8,
    onClick: () => console.log("Card clicked"),
  },
};

export const TwoLineNameLongDescription: Story = {
  args: {
    agentName: "Advanced Natural Language Processing and Machine Learning",
    subHeading: "State-of-the-art NLP & ML",
    description:
      "Cutting-edge natural language processing and machine learning capabilities combined in one powerful tool. Perfect for researchers and developers working on advanced AI applications.",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorName: "AI Research Labs",
    runs: 150000,
    rating: 4.9,
    onClick: () => console.log("Card clicked"),
  },
};

export const WithInteraction: Story = {
  args: {
    agentName: "AI Writing Assistant",
    subHeading: "Enhance your writing",
    description:
      "An AI-powered writing assistant that helps improve your writing style and clarity.",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorName: "WordCraft AI",
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
