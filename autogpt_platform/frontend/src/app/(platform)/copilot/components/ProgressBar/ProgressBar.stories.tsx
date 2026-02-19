import type { Meta, StoryObj } from "@storybook/nextjs";
import { ProgressBar } from "./ProgressBar";

const meta: Meta<typeof ProgressBar> = {
  title: "CoPilot/Loaders/ProgressBar",
  component: ProgressBar,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Horizontal progress bar with label and percentage indicator.",
      },
    },
  },
  decorators: [
    (Story) => (
      <div className="w-80">
        <Story />
      </div>
    ),
  ],
};
export default meta;
type Story = StoryObj<typeof ProgressBar>;

export const Empty: Story = {
  args: { value: 0 },
};

export const Partial: Story = {
  args: { value: 42 },
};

export const Full: Story = {
  args: { value: 100 },
};

export const CustomLabel: Story = {
  args: { value: 65, label: "Uploading files..." },
};

export const OverBounds: Story = {
  args: { value: 150 },
};
