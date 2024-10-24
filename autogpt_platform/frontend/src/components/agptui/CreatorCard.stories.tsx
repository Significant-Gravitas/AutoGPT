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
    avatarSrc: { control: "text" },
  },
} satisfies Meta<typeof CreatorCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    creatorName: "John Doe",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    bio: "AI enthusiast and developer with a passion for creating innovative agents.",
    agentsUploaded: 15,
    onClick: () => console.log("Default CreatorCard clicked"),
    avatarSrc: "https://github.com/shadcn.png",
  },
};

export const NewCreator: Story = {
  args: {
    creatorName: "Jane Smith",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    bio: "Excited to start my journey in AI agent development!",
    agentsUploaded: 1,
    onClick: () => console.log("NewCreator CreatorCard clicked"),
    avatarSrc: "https://example.com/avatar2.jpg",
  },
};

export const ExperiencedCreator: Story = {
  args: {
    creatorName: "Alex Johnson",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    bio: "Veteran AI researcher with a focus on natural language processing and machine learning.",
    agentsUploaded: 50,
    onClick: () => console.log("ExperiencedCreator CreatorCard clicked"),
    avatarSrc: "https://example.com/avatar3.jpg",
  },
};

export const WithInteraction: Story = {
  args: {
    creatorName: "Sam Brown",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    bio: "Exploring the frontiers of AI and its applications in everyday life.",
    agentsUploaded: 30,
    onClick: () => console.log("WithInteraction CreatorCard clicked"),
    avatarSrc: "https://example.com/avatar4.jpg",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const creatorCard = canvas.getByText("Sam Brown");

    await userEvent.hover(creatorCard);
    await userEvent.click(creatorCard);
  },
};
