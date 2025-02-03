import type { Meta, StoryObj } from "@storybook/react";
import { CreatorCard } from "./CreatorCard";
import { userEvent, within } from "@storybook/test";

const meta = {
  title: "AGPT UI/Creator Card",
  component: CreatorCard,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    creatorName: { control: "text" },
    creatorImage: { control: "text" },
    bio: { control: "text" },
    agentsUploaded: { control: "number" },
    onClick: { action: "clicked" },
  },
} satisfies Meta<typeof CreatorCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    index: 0,
    creatorName: "John Doe",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    bio: "AI enthusiast and developer with a passion for creating innovative agents.",
    agentsUploaded: 15,
    onClick: () => console.log("Default CreatorCard clicked"),
  },
};

export const NewCreator: Story = {
  args: {
    index: 1,
    creatorName: "Jane Smith",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    bio: "Excited to start my journey in AI agent development!",
    agentsUploaded: 1,
    onClick: () => console.log("NewCreator CreatorCard clicked"),
  },
};

export const ExperiencedCreator: Story = {
  args: {
    index: 2,
    creatorName: "Alex Johnson",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    bio: "Veteran AI researcher with a focus on natural language processing and machine learning.",
    agentsUploaded: 50,
    onClick: () => console.log("ExperiencedCreator CreatorCard clicked"),
  },
};

export const WithInteraction: Story = {
  args: {
    index: 3,
    creatorName: "Sam Brown",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    bio: "Exploring the frontiers of AI and its applications in everyday life.",
    agentsUploaded: 30,
    onClick: () => console.log("WithInteraction CreatorCard clicked"),
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const creatorCard = canvas.getByText("Sam Brown");

    await userEvent.hover(creatorCard);
    await userEvent.click(creatorCard);
  },
};
