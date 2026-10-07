import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { TrialTitle } from "./TrialTitle";

const meta = {
  title: "Organisms/TrialTitle",
  component: TrialTitle,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          "The heading used by every trial surface (offer, status and rejection). Renders an h2.",
      },
    },
  },
  args: { children: "Try AutoGPT Pro for 7 days" },
} satisfies Meta<typeof TrialTitle>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Status: Story = {
  args: { children: "Your trial has ended" },
};

export const LongTitle: Story = {
  decorators: [
    (Story) => (
      <div className="w-96">
        <Story />
      </div>
    ),
  ],
  args: { children: "This introductory offer has already been used" },
};
