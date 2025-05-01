import type { Meta, StoryObj } from "@storybook/react";
import { CreatorInfoCard } from "./CreatorInfoCard";

const meta = {
  title: "new/Creator Info Card",
  component: CreatorInfoCard,
  decorators: [
    (Story) => (
      <div className="flex h-screen w-screen items-center justify-center p-4">
        <Story />
      </div>
    ),
  ],
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

export const LongContent: Story = {
  args: {
    username: "This Is An Extremel Long Username To Test",
    handle: "this_is_an_extremely_long_there_what",
    avatarSrc: "https://example.com/avatar2.jpg",
    categories: [
      "Artificial Intelligence",
      "Machine Learning",
      "Neural Networks",
      "Deep Learning",
      "Natural Language Processing",
      "Computer Vision",
      "Robotics",
      "Data Science",
      "Cloud Computing",
      "Internet of Things",
    ],
    averageRating: 4.8888888888,
    totalRuns: 1000000000,
  },
};
