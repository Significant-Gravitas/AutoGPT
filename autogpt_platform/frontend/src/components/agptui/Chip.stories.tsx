import type { Meta, StoryObj } from "@storybook/react";
import { Chip } from "./Chip";

const meta: Meta<typeof Chip> = {
  component: Chip,
  title: "new/BasicBadge",
  argTypes: {
    children: {
      control: "text",
      description: "The content of the badge",
    },
  },
  parameters: {
    layout: "centered",
  },
};

export default meta;
type Story = StoryObj<typeof Chip>;

export const Default: Story = {
  args: {
    children: "Marketing",
  },
};
