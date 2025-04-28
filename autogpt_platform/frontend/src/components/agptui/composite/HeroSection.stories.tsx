import type { Meta, StoryObj } from "@storybook/react";
import { HeroSection } from "./HeroSection";
import { userEvent, within, expect } from "@storybook/test";

const meta = {
  title: "AGPT UI/Composite/Hero Section",
  component: HeroSection,
  decorators: [
    (Story) => (
      <div className="flex items-center justify-center py-4 md:p-4">
        <Story />
      </div>
    ),
  ],
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
    const searchInput = canvas.getByRole("textbox");

    await userEvent.type(searchInput, "test query");
    await userEvent.keyboard("{Enter}");

    await expect(searchInput).toHaveValue("test query");

    const filterChip = canvas.getByText("Marketing");
    await userEvent.click(filterChip);
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
    const searchInput = canvas.getByRole("textbox");

    await userEvent.click(searchInput);
    await userEvent.keyboard("{Enter}");

    await expect(searchInput).toHaveValue("");
  },
};

export const FilterInteraction: Story = {
  args: {
    onSearch: (query: string) => console.log(`Searched: ${query}`),
    onFilterChange: (selectedFilters: string[]) =>
      console.log(`Filters changed: ${selectedFilters.join(", ")}`),
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);

    const filterChips = canvas.getAllByTestId("filter-chip");

    for (const chip of filterChips) {
      await userEvent.click(chip);
    }
  },
};

export const MobileView: Story = {
  parameters: {
    viewport: {
      defaultViewport: "mobile2",
    },
  },
  args: {
    onSearch: (query: string) => console.log(`Searched: ${query}`),
    onFilterChange: (selectedFilters: string[]) =>
      console.log(`Filters changed: ${selectedFilters.join(", ")}`),
  },
};

export const TabletView: Story = {
  parameters: {
    viewport: {
      defaultViewport: "tablet",
    },
  },
  args: {
    onSearch: (query: string) => console.log(`Searched: ${query}`),
    onFilterChange: (selectedFilters: string[]) =>
      console.log(`Filters changed: ${selectedFilters.join(", ")}`),
  },
};
