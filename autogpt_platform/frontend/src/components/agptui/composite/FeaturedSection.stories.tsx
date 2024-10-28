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
    agentName: "SEO Optimizer Pro",
    subHeading: "Optimize your website's SEO",
    creatorName: "AI Solutions Inc.",
    description:
      "Boost your website's search engine rankings with our advanced AI-powered SEO optimization tool.",
    runs: 50000,
    rating: 4.7,
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
  },
  {
    agentName: "Content Writer AI",
    subHeading: "Create engaging content",
    creatorName: "WordCraft AI",
    description:
      "Generate high-quality, engaging content for your blog, social media, or marketing campaigns.",
    runs: 75000,
    rating: 4.5,
    agentImage:
      "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
  },
  {
    agentName: "Data Analyzer Lite",
    subHeading: "Analyze data with ease",
    creatorName: "DataTech",
    description: "A basic tool for analyzing small to medium-sized datasets.",
    runs: 10000,
    rating: 3.8,
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
  },
  {
    agentName: "Social Media Manager",
    subHeading: "Automate your social presence",
    creatorName: "SocialBot Inc",
    description:
      "Schedule posts, analyze engagement, and grow your social media following.",
    runs: 25000,
    rating: 4.2,
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
  },
  {
    agentName: "Email Marketing Assistant",
    subHeading: "Optimize email campaigns",
    creatorName: "EmailPro AI",
    description:
      "Create and optimize email campaigns with AI-powered suggestions.",
    runs: 30000,
    rating: 4.4,
    agentImage:
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
    const firstCard = canvas.getAllByRole("featured-store-card")[0];
    await userEvent.click(firstCard);
    await userEvent.hover(firstCard);
    await expect(firstCard).toHaveClass("hover:shadow-lg");
  },
};
