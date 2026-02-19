import type { Meta, StoryObj } from "@storybook/nextjs";
import { OrbitLoader } from "./OrbitLoader";

const meta: Meta<typeof OrbitLoader> = {
  title: "CoPilot/Loaders/OrbitLoader",
  component: OrbitLoader,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component: "Animated orbiting ball loader used during CoPilot actions.",
      },
    },
  },
};
export default meta;
type Story = StoryObj<typeof OrbitLoader>;

export const Default: Story = {};

export const Small: Story = {
  args: { size: 16 },
};

export const Large: Story = {
  args: { size: 48 },
};
