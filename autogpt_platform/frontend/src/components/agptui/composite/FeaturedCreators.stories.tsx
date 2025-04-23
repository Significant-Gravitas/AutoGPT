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
    // onCardClick: { action: "cardClicked" },
  },
} satisfies Meta<typeof FeaturedCreators>;

export default meta;
type Story = StoryObj<typeof meta>;

const defaultCreators = [
  {
    name: "AI Innovator",
    username: "ai_innovator",
    description:
      "Pushing the boundaries of AI technology with cutting-edge solutions and innovative approaches to machine learning.",
    avatar_url:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    num_agents: 15,
  },
  {
    name: "Code Wizard",
    username: "code_wizard",
    description:
      "Crafting elegant solutions with AI and helping others learn the magic of coding.",
    avatar_url:
      "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
    num_agents: 8,
  },
  {
    name: "Data Alchemist",
    username: "data_alchemist",
    description:
      "Transforming raw data into AI gold. Specializing in data processing and analytics.",
    avatar_url:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    num_agents: 12,
  },
  {
    name: "ML Master",
    username: "ml_master",
    description:
      "Specializing in machine learning algorithms and neural network architectures.",
    avatar_url:
      "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
    num_agents: 20,
  },
];

export const Default: Story = {
  args: {
    featuredCreators: defaultCreators,
    // onCardClick: (creatorName) => console.log(`Clicked on ${creatorName}`),
  },
};

export const SingleCreator: Story = {
  args: {
    featuredCreators: [defaultCreators[0]],
    // onCardClick: (creatorName) => console.log(`Clicked on ${creatorName}`),
  },
};

export const ManyCreators: Story = {
  args: {
    featuredCreators: [
      ...defaultCreators,
      {
        name: "NLP Ninja",
        username: "nlp_ninja",
        description:
          "Expert in natural language processing and text analysis systems.",
        avatar_url:
          "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
        num_agents: 18,
      },
      {
        name: "AI Explorer",
        username: "ai_explorer",
        description:
          "Discovering new frontiers in artificial intelligence and autonomous systems.",
        avatar_url:
          "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
        num_agents: 25,
      },
    ],
    // onCardClick: (creatorName) => console.log(`Clicked on ${creatorName}`),
  },
};

export const WithInteraction: Story = {
  args: {
    featuredCreators: defaultCreators,
    // onCardClick: (creatorName) => console.log(`Clicked on ${creatorName}`),
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
