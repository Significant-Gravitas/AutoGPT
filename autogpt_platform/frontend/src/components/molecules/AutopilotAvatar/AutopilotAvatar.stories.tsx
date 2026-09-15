import type { Meta, StoryObj } from "@storybook/nextjs";
import { AutopilotAvatar } from "./AutopilotAvatar";

const meta = {
  title: "Molecules/AutopilotAvatar",
  component: AutopilotAvatar,
  parameters: { layout: "centered" },
  tags: ["autodocs"],
} satisfies Meta<typeof AutopilotAvatar>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Sizes: Story = {
  render: function SizesStory() {
    return (
      <div className="flex items-center gap-4">
        {[18, 20, 24, 36].map((size) => (
          <AutopilotAvatar key={size} size={size} />
        ))}
      </div>
    );
  },
};

export const SquareChip: Story = {
  args: { size: 36, className: "rounded-xl" },
};
