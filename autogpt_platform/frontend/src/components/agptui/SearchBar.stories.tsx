import type { Meta, StoryObj } from "@storybook/react";
import { SearchBar } from "./SearchBar";
import { userEvent, within, expect } from "@storybook/test";

const meta = {
  title: "AGPT UI/Search Bar",
  component: SearchBar,
  parameters: {
    nextjs: {
      appDirectory: true,
      navigation: {
        pathname: "/marketplace/search",
        query: {
          searchTerm: "",
        },
      },
    },
  },
  tags: ["autodocs"],
  argTypes: {
    placeholder: { control: "text" },
    backgroundColor: { control: "text" },
    iconColor: { control: "text" },
    textColor: { control: "text" },
    placeholderColor: { control: "text" },
    width: { control: "text" },
    height: { control: "text" },
  },
  decorators: [
    (Story) => (
      <div className="mx-auto w-full max-w-screen-lg p-4">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof SearchBar>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    placeholder: 'Search for tasks like "optimise SEO"',
  },
};

export const CustomStyles: Story = {
  args: {
    placeholder: "Enter your search query",
    backgroundColor: "bg-blue-100",
    iconColor: "text-blue-500",
    textColor: "text-blue-700",
    placeholderColor: "text-blue-400",
  },
};

export const CustomDimensions: Story = {
  args: {
    placeholder: "Custom size search bar",
    width: "w-full md:w-[30rem]",
    height: "h-[45px]",
  },
};

export const DarkMode: Story = {
  args: {
    placeholder: "Dark mode search",
    backgroundColor: "bg-neutral-800",
    iconColor: "text-neutral-400",
    textColor: "text-neutral-200",
    placeholderColor: "text-neutral-400",
  },
  parameters: {
    backgrounds: { default: "dark" },
  },
};

export const WithInteraction: Story = {
  args: {
    placeholder: "Type and press Enter",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const input = canvas.getByTestId("store-search-input");

    await userEvent.type(input, "test query");
    await userEvent.keyboard("{Enter}");

    await expect(input).toHaveValue("test query");
  },
};

export const EmptySubmit: Story = {
  args: {
    placeholder: "Empty submit test",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const input = canvas.getByTestId("store-search-input");

    await userEvent.click(input);
    await userEvent.keyboard("{Enter}");

    await expect(input).toHaveValue("");
  },
};

export const MobileViewCompact: Story = {
  args: {
    placeholder: "Search on mobile",
    width: "w-full",
    height: "h-[45px]",
  },
  parameters: {
    viewport: {
      defaultViewport: "mobile1",
    },
  },
};

export const ExtraLongPlaceholder: Story = {
  args: {
    placeholder:
      "This is an extremely long placeholder text that demonstrates how the search bar handles overflow with very long placeholder text",
  },
};
