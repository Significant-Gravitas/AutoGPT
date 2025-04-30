import type { Meta, StoryObj } from "@storybook/react";
import { CreatorCard } from "./CreatorCard";
import { userEvent, within, expect } from "@storybook/test";

const meta = {
  title: "AGPT UI/Creator Card",
  component: CreatorCard,
  decorators: [
    (Story) => (
      <div className="flex items-center justify-center p-4">
        <Story />
      </div>
    ),
  ],
  tags: ["autodocs"],
  argTypes: {
    creatorName: { control: "text" },
    creatorImage: { control: "text" },
    bio: { control: "text" },
    agentsUploaded: { control: "number" },
    onClick: { action: "clicked" },
    index: { control: "number" },
  },
} satisfies Meta<typeof CreatorCard>;

export default meta;
type Story = StoryObj<typeof meta>;

const defaultAvatarImage = "/default_avatar.png";

export const Default: Story = {
  args: {
    index: 0,
    creatorName: "John Doe",
    creatorImage: defaultAvatarImage,
    bio: "AI enthusiast and developer with a passion for creating innovative agents.",
    agentsUploaded: 15,
    onClick: () => console.log("Default CreatorCard clicked"),
  },
};

export const NoImage: Story = {
  args: {
    index: 0,
    creatorName: "John Doe",
    creatorImage: "",
    bio: "AI enthusiast and developer with a passion for creating innovative agents.",
    agentsUploaded: 15,
    onClick: () => console.log("NoImage CreatorCard clicked"),
  },
};

export const LongContent: Story = {
  args: {
    index: 1,
    creatorName: "Alexandria Rodriguez-Fitzgerald Johnson III",
    creatorImage: defaultAvatarImage,
    bio: "Excited to start my journey in AI agent development! I have a background in computer science and machine learning, with a special interest in creating agents that can assist with everyday tasks and solve complex problems efficiently.",
    agentsUploaded: 500000,
    onClick: () => console.log("LongName CreatorCard clicked"),
  },
};

export const TestingInteractions: Story = {
  args: {
    index: 3,
    creatorName: "Sam Brown",
    creatorImage: defaultAvatarImage,
    bio: "Exploring the frontiers of AI and its applications in everyday life.",
    agentsUploaded: 30,
    onClick: () => console.log("WithInteraction CreatorCard clicked"),
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const creatorCard = canvas.getByTestId("creator-card");

    // Test hover state
    await userEvent.hover(creatorCard);
    await new Promise((resolve) => setTimeout(resolve, 500));

    // Test click interaction
    await userEvent.click(creatorCard);
  },
};
