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
  "Lorem ipsum",
  "Lorem ipsum",
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

export const WithSelectedFilters: Story = {
  args: {
    badges: defaultBadges,
    multiSelect: true,
  },
  play: async ({ canvasElement, args }) => {
    const canvas = within(canvasElement);
    const marketingChip = canvas.getByText("Marketing").parentElement;
    const salesChip = canvas.getByText("Sales").parentElement;
    if (!marketingChip || !salesChip) {
      throw new Error("Marketing or Sales chip not found");
    }

    await userEvent.click(marketingChip);
    await userEvent.click(salesChip);

    await expect(marketingChip).toHaveClass("bg-neutral-100");
    await expect(salesChip).toHaveClass("bg-neutral-100");
  },
};

export const WithFilterChangeCallback: Story = {
  args: {
    badges: defaultBadges,
    multiSelect: true,
    onFilterChange: (selectedFilters: string[]) => {
      console.log("Selected filters:", selectedFilters);
    },
  },
  play: async ({ canvasElement, args }) => {
    const canvas = within(canvasElement);
    const salesChip = canvas.getByText("Sales");
    const marketingChip = canvas.getByText("Marketing");

    await userEvent.click(salesChip);
    await userEvent.click(marketingChip);
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
      "Natural Language Processing",
      "Computer Vision",
      "Data Science",
    ],
    multiSelect: true,
  },
};

export const SingleSelectBehavior: Story = {
  args: {
    badges: defaultBadges,
    multiSelect: false,
  },
  play: async ({ canvasElement, args }) => {
    const canvas = within(canvasElement);
    const salesChip = canvas.getByText("Sales").parentElement;
    const marketingChip = canvas.getByText("Marketing").parentElement;

    if (!salesChip || !marketingChip) {
      throw new Error("Sales or Marketing chip not found");
    }

    await userEvent.click(salesChip);
    await expect(salesChip).toHaveClass("bg-neutral-100");

    await userEvent.click(marketingChip);
    await expect(marketingChip).toHaveClass("bg-neutral-100");
    await expect(salesChip).not.toHaveClass("bg-neutral-100");
  },
};
