import type { Meta, StoryObj } from "@storybook/react";
import { CreatorInfoCard } from "./CreatorInfoCard";

const meta = {
  title: "AGPT UI/Creator Info Card",
  component: CreatorInfoCard,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    username: { control: "text" },
    handle: { control: "text" },
    avatarSrc: { control: "text" },
    categories: { control: "object" },
    averageRating: { control: "number", min: 0, max: 5, step: 0.1 },
    totalRuns: { control: "number" },
  },
} satisfies Meta<typeof CreatorInfoCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    username: "SignificantGravitas",
    handle: "oliviagrace1421",
    avatarSrc: "https://github.com/shadcn.png",
    categories: ["Entertainment", "Business"],
    averageRating: 4.7,
    totalRuns: 1500,
  },
};

export const NewCreator: Story = {
  args: {
    username: "AI Enthusiast",
    handle: "ai_newbie",
    avatarSrc: "https://example.com/avatar2.jpg",
    categories: ["AI", "Technology"],
    averageRating: 0,
    totalRuns: 0,
  },
};

export const ExperiencedCreator: Story = {
  args: {
    username: "Tech Master",
    handle: "techmaster",
    avatarSrc: "https://example.com/avatar3.jpg",
    categories: ["AI", "Development", "Education"],
    averageRating: 4.9,
    totalRuns: 50000,
  },
};
