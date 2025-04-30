import type { Meta, StoryObj } from "@storybook/react";
import { FilterChips } from "./FilterChips";
import { userEvent, within, expect } from "@storybook/test";

const meta = {
  title: "AGPT UI/Filter Chips",
  component: FilterChips,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    badges: { control: "object" },
    onFilterChange: { action: "onFilterChange" },
    multiSelect: { control: "boolean" },
  },
} satisfies Meta<typeof FilterChips>;

export default meta;
type Story = StoryObj<typeof meta>;

const defaultBadges = [
  "Marketing",
  "Sales",
  "Content creation",
  "AI",
  "Data Science",
];

export const Default: Story = {
  args: {
    badges: defaultBadges,
    multiSelect: true,
  },
};

export const SingleSelect: Story = {
  args: {
    badges: defaultBadges,
    multiSelect: false,
  },
};

export const EmptyBadges: Story = {
  args: {
    badges: [],
    multiSelect: true,
  },
};

export const LongBadgeNames: Story = {
  args: {
    badges: [
      "Machine Learning",
      "Natural Language Processing, Natural Language Processing, Natural Language Processing",
      "Computer Vision",
      "Data Science",
    ],
    multiSelect: true,
  },
};
