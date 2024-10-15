import type { Meta, StoryObj } from "@storybook/react";
import { HeroSection } from "./HeroSection";
import { userEvent, within, expect } from "@storybook/test";

const meta = {
  title: "AGPTUI/Marketplace/Home/HeroSection",
  component: HeroSection,
  parameters: {
    layout: {
      center: true,
      fullscreen: true,
      padding: 0,
    },
  },
  tags: ["autodocs"],
  argTypes: {
    onSearch: { action: "searched" },
    onFilterChange: { action: "filtersChanged" },
  },
} satisfies Meta<typeof HeroSection>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    onSearch: (query: string) => console.log(`Searched: ${query}`),
    onFilterChange: (selectedFilters: string[]) =>
      console.log(`Filters changed: ${selectedFilters.join(", ")}`),
  },
};

export const WithInteraction: Story = {
  args: {
    onSearch: (query: string) => console.log(`Searched: ${query}`),
    onFilterChange: (selectedFilters: string[]) =>
      console.log(`Filters changed: ${selectedFilters.join(", ")}`),
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const searchInput = canvas.getByPlaceholderText(/Search for tasks/i);

    await userEvent.type(searchInput, "test query");
    await userEvent.keyboard("{Enter}");

    await expect(searchInput).toHaveValue("test query");

    const filterChip = canvas.getByText("Marketing");
    await userEvent.click(filterChip);

    await expect(filterChip).toHaveClass("bg-[#F1F1F1]");
  },
};

export const EmptySearch: Story = {
  args: {
    onSearch: (query: string) => console.log(`Searched: ${query}`),
    onFilterChange: (selectedFilters: string[]) =>
      console.log(`Filters changed: ${selectedFilters.join(", ")}`),
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const searchInput = canvas.getByPlaceholderText(/Search for tasks/i);

    await userEvent.keyboard("{Enter}");

    await expect(searchInput).toHaveValue("");
  },
};
