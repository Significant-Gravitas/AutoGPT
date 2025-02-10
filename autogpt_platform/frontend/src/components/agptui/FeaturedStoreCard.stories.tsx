import type { Meta, StoryObj } from "@storybook/react";
import { FeaturedAgentCard } from "./FeaturedAgentCard";
import { userEvent, within } from "@storybook/test";

const meta = {
  title: "AGPT UI/Featured Store Card",
  component: FeaturedAgentCard,
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
  },
} satisfies Meta<typeof FeaturedAgentCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    agent: {
      agent_name: "Personalized Morning Coffee Newsletter example of three lines",
      sub_heading:
          "Transform ideas into breathtaking images with this AI-powered Image Generator.",
      description:
          "Elevate your web content with this powerful AI Webpage Copy Improver. Designed for marketers, SEO specialists, and web developers, this tool analyses and enhances website copy for maximum impact. Using advanced language models, it optimizes text for better clarity, SEO performance, and increased conversion rates.",
      agent_image:
          "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
      creator_avatar:
          "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
      creator: "AI Solutions Inc.",
      runs: 50000,
      rating: 4.7,
      slug: ""
    },
    backgroundColor: "bg-white",
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
    backgroundColor: "bg-white",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const featuredCard = canvas.getByTestId("featured-store-card");
    await userEvent.hover(featuredCard);
    await userEvent.click(featuredCard);
  },
};
