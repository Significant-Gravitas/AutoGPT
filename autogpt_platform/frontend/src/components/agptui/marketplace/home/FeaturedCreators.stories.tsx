import type { Meta, StoryObj } from "@storybook/react";
import { FeaturedCreators } from "./FeaturedCreators";
import { userEvent, within } from "@storybook/test";

const meta = {
  title: "AGPTUI/Marketplace/Home/FeaturedCreators",
  component: FeaturedCreators,
  parameters: {
    layout: "centered",
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
        creatorName: "ML Master",
        creatorImage:
          "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
        bio: "Specializing in machine learning algorithms",
        agentsUploaded: 20,
        avatarSrc: "https://example.com/avatar4.jpg",
      },
      {
        creatorName: "NLP Ninja",
        creatorImage:
          "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
        bio: "Expert in natural language processing",
        agentsUploaded: 18,
        avatarSrc: "https://example.com/avatar5.jpg",
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
    const firstCreatorCard = canvas.getByText("AI Innovator");

    await userEvent.hover(firstCreatorCard);
    await userEvent.click(firstCreatorCard);
  },
};
