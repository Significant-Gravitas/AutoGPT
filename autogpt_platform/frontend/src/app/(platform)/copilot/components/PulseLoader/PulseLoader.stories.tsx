import type { Meta, StoryObj } from "@storybook/nextjs";
import { PulseLoader } from "./PulseLoader";

const meta: Meta<typeof PulseLoader> = {
  title: "CoPilot/Loaders/PulseLoader",
  component: PulseLoader,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component: "Pulsing circle loader animation.",
      },
    },
  },
};
export default meta;
type Story = StoryObj<typeof PulseLoader>;

export const Default: Story = {};

export const Small: Story = {
  args: { size: 16 },
};

export const Large: Story = {
  args: { size: 48 },
};
