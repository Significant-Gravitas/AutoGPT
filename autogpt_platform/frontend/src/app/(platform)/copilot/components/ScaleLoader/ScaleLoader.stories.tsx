import type { Meta, StoryObj } from "@storybook/nextjs";
import { ScaleLoader } from "./ScaleLoader";

const meta: Meta<typeof ScaleLoader> = {
  title: "CoPilot/Loaders/ScaleLoader",
  component: ScaleLoader,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component: "Scaling bar loader animation.",
      },
    },
  },
};
export default meta;
type Story = StoryObj<typeof ScaleLoader>;

export const Default: Story = {};

export const Small: Story = {
  args: { size: 24 },
};

export const Large: Story = {
  args: { size: 72 },
};
