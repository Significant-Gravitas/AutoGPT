import type { Meta, StoryObj } from "@storybook/react";
import { SearchBar } from "./SearchBar";
import { userEvent, within, expect } from "@storybook/test";

const meta = {
  title: "Agpt Custom ui/marketing/Search Bar",
  component: SearchBar,

  tags: ["autodocs"],
  argTypes: {
    placeholder: { control: "text" },
    className: { control: "text" },
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
    className: "bg-blue-100",
  },
};

export const TestingInteractions: Story = {
  args: {
    placeholder: "Type and press Enter",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);

    // checking onChange in input
    const Input = canvas.getByTestId("store-search-input");
    await userEvent.type(Input, "test query", {
      delay: 100,
    });
    await userEvent.keyboard("{Enter}", {
      delay: 100,
    });
    await expect(Input).toHaveValue("test query");
  },
};
