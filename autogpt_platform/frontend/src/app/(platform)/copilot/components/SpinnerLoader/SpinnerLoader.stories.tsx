import type { Meta, StoryObj } from "@storybook/nextjs";
import { SpinnerLoader } from "./SpinnerLoader";

const meta: Meta<typeof SpinnerLoader> = {
  title: "CoPilot/Loaders/SpinnerLoader",
  component: SpinnerLoader,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component: "Classic spinning loader animation.",
      },
    },
  },
};
export default meta;
type Story = StoryObj<typeof SpinnerLoader>;

export const Default: Story = {};

export const Small: Story = {
  args: { size: 16 },
};

export const Large: Story = {
  args: { size: 48 },
};
