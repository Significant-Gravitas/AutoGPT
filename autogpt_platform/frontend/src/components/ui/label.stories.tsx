import type { Meta, StoryObj } from "@storybook/react";

import { Label } from "./label";

const meta = {
  title: "UI/Label",
  component: Label,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    htmlFor: { control: "text" },
  },
} satisfies Meta<typeof Label>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    children: "Default Label",
  },
};

export const WithHtmlFor: Story = {
  args: {
    htmlFor: "example-input",
    children: "Label with htmlFor",
  },
};

export const CustomContent: Story = {
  args: {
    children: (
      <>
        <span className="mr-1">📝</span>
        Custom Label Content
      </>
    ),
  },
};

export const WithClassName: Story = {
  args: {
    className: "text-blue-500 font-bold",
    children: "Styled Label",
  },
};
