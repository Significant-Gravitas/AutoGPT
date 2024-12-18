import type { Meta, StoryObj } from "@storybook/react";
import { FeaturedSection } from "./FeaturedSection";
import { userEvent, within, expect } from "@storybook/test";

const meta = {
  title: "AGPT UI/Composite/Featured Agents",
  component: FeaturedSection,
  parameters: {
    layout: {
      center: true,
      fullscreen: true,
      padding: 0,
    },
  },
  tags: ["autodocs"],
  argTypes: {
    featuredAgents: { control: "object" },
    onCardClick: { action: "clicked" },
  },
} satisfies Meta<typeof FeaturedSection>;

export default meta;
type Story = StoryObj<typeof meta>;

const mockFeaturedAgents = [
  {
    agentName: "Personalized Morning Coffee Newsletter example of three lines",
    subHeading:
      "Transform ideas into breathtaking images with this AI-powered Image Generator.",
    creatorName: "AI Solutions Inc.",
    description:
      "Elevate your web content with this powerful AI Webpage Copy Improver. Designed for marketers, SEO specialists, and web developers, this tool analyses and enhances website copy for maximum impact. Using advanced language models, it optimizes text for better clarity, SEO performance, and increased conversion rates.",
    runs: 50000,
    rating: 4.7,
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
  },
  {
    agentName: "Data Analyzer Lite",
    subHeading: "Basic data analysis tool",
    creatorName: "DataTech",
    description:
      "A lightweight data analysis tool for basic data processing needs.",
    runs: 10000,
    rating: 2.8,
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
  },
  {
    agentName: "CodeAssist AI",
    subHeading: "Your AI coding companion",
    creatorName: "DevTools Co.",
    description:
      "An intelligent coding assistant that helps developers write better code faster.",
    runs: 1000000,
    rating: 4.9,
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
  },
  {
    agentName: "MultiTasker",
    subHeading: "All-in-one productivity suite",
    creatorName: "Productivity Plus",
    description:
      "A comprehensive productivity suite that combines task management, note-taking, and project planning into one seamless interface. Features include smart task prioritization, automated scheduling, and AI-powered insights to help you work more efficiently.",
    runs: 75000,
    rating: 4.5,
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
  },
  {
    agentName: "QuickTask",
    subHeading: "Fast task automation",
    creatorName: "EfficientWorks",
    description: "Simple and efficient task automation tool.",
    runs: 50000,
    rating: 4.2,
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
  },
];

export const Default: Story = {
  args: {
    featuredAgents: mockFeaturedAgents,
    onCardClick: (agentName: string) => console.log(`Clicked on ${agentName}`),
  },
};

export const SingleAgent: Story = {
  args: {
    featuredAgents: [mockFeaturedAgents[0]],
    onCardClick: (agentName: string) => console.log(`Clicked on ${agentName}`),
  },
};

export const NoAgents: Story = {
  args: {
    featuredAgents: [],
    onCardClick: (agentName: string) => console.log(`Clicked on ${agentName}`),
  },
};

export const WithInteraction: Story = {
  args: {
    featuredAgents: mockFeaturedAgents,
    onCardClick: (agentName: string) => console.log(`Clicked on ${agentName}`),
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const featuredCard = canvas.getByText(
      "Personalized Morning Coffee Newsletter example of three lines",
    );

    await userEvent.hover(featuredCard);
    await userEvent.click(featuredCard);
  },
};
