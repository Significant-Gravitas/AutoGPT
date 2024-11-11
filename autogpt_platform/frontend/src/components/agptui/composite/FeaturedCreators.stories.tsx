import type { Meta, StoryObj } from "@storybook/react";
import { FeaturedCreators } from "./FeaturedCreators";
import { userEvent, within, expect } from "@storybook/test";

const meta = {
  title: "AGPT UI/Composite/Featured Creators",
  component: FeaturedCreators,
  parameters: {
    layout: {
      center: true,
      fullscreen: true,
      padding: 0,
    },
  },
  tags: ["autodocs"],
  argTypes: {
    featuredCreators: { control: "object" },
    onCardClick: { action: "cardClicked" },
  },
} satisfies Meta<typeof FeaturedCreators>;

export default meta;
type Story = StoryObj<typeof meta>;

const defaultCreators = [
  {
    creatorName: "AI Innovator",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    bio: "Pushing the boundaries of AI technology",
    agentsUploaded: 15,
    avatarSrc: "https://example.com/avatar1.jpg",
  },
  {
    creatorName: "Code Wizard",
    creatorImage:
      "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
    bio: "Crafting elegant solutions with AI",
    agentsUploaded: 8,
    avatarSrc: "https://example.com/avatar2.jpg",
  },
  {
    creatorName: "Data Alchemist",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    bio: "Transforming raw data into AI gold",
    agentsUploaded: 12,
    avatarSrc: "https://example.com/avatar3.jpg",
  },
];

export const Default: Story = {
  args: {
    featuredCreators: defaultCreators,
    onCardClick: (creatorName) => console.log(`Clicked on ${creatorName}`),
  },
};

export const SingleCreator: Story = {
  args: {
    featuredCreators: [defaultCreators[0]],
    onCardClick: (creatorName) => console.log(`Clicked on ${creatorName}`),
  },
};

export const ManyCreators: Story = {
  args: {
    featuredCreators: [
      ...defaultCreators,
      {
        name: "ML Master",
        avatar_url:
          "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
        description: "Specializing in machine learning algorithms",
        num_agents: 20,
        avatar_url: "https://example.com/avatar4.jpg",
      },
      {
        name: "NLP Ninja",
        avatar_url:
          "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
        description: "Expert in natural language processing",
        num_agents: 18,
        avatar_url: "https://example.com/avatar5.jpg",
      },
    ],
    onCardClick: (creatorName) => console.log(`Clicked on ${creatorName}`),
  },
};

export const WithInteraction: Story = {
  args: {
    featuredCreators: defaultCreators,
    onCardClick: (creatorName) => console.log(`Clicked on ${creatorName}`),
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const creatorCards = canvas.getAllByRole("creator-card");
    const firstCreatorCard = creatorCards[0];

    await userEvent.hover(firstCreatorCard);
    await userEvent.click(firstCreatorCard);

    // Check if the card has the expected hover and click effects
    await expect(firstCreatorCard).toHaveClass("hover:shadow-lg");
  },
};
