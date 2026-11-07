import type { Meta, StoryObj } from "@storybook/nextjs";
import { YouTubeEmbed } from "./YouTubeEmbed";

const meta = {
  title: "Molecules/YouTubeEmbed",
  component: YouTubeEmbed,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="aspect-video w-full max-w-2xl overflow-hidden rounded-lg">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof YouTubeEmbed>;

export default meta;
type Story = StoryObj<typeof meta>;

// Storybook has no consent banner, so this shows the notice with "Load video".
export const WithoutConsent: Story = {
  args: {
    url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    title: "Demo video",
  },
};
