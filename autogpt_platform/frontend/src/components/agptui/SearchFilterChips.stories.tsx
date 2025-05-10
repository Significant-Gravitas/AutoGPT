import type { Meta, StoryObj } from "@storybook/react";
import { SearchFilterChips } from "./SearchFilterChips";

const meta = {
  title: "Agpt Custom UI/marketing/Search Filter Chips",
  component: SearchFilterChips,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    onFilterChange: { action: "onFilterChange" },
  },
} satisfies Meta<typeof SearchFilterChips>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};
