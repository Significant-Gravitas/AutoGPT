import type { Meta, StoryObj } from "@storybook/nextjs";
import { AITeamIcon } from "./AITeamIcon";

const meta: Meta<typeof AITeamIcon> = {
  title: "Atoms/AITeamIcon",
  tags: ["autodocs"],
  component: AITeamIcon,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Brand mark for the AI team surfaces (marketplace experts section and the Team page). It inherits the current text color, so set the color on a parent or via className.",
      },
    },
  },
  argTypes: {
    size: {
      control: { type: "number" },
      description: "Width and height in pixels.",
    },
    className: { control: { type: "text" } },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { size: 36 },
};

export const Sizes: Story = {
  render: renderSizes,
};

export const Colored: Story = {
  args: { size: 48, className: "text-violet-600" },
};

function renderSizes() {
  return (
    <div className="flex items-end gap-4">
      {[16, 24, 36, 64].map((size) => (
        <AITeamIcon key={size} size={size} />
      ))}
    </div>
  );
}
