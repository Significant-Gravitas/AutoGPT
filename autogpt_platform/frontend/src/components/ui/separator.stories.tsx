import React from "react";
import type { Meta, StoryObj } from "@storybook/react";

import { Separator } from "./separator";

const meta = {
  title: "UI/Separator",
  component: Separator,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    orientation: {
      control: "select",
      options: ["horizontal", "vertical"],
    },
    className: { control: "text" },
  },
} satisfies Meta<typeof Separator>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {},
};

export const Horizontal: Story = {
  args: {
    orientation: "horizontal",
  },
  decorators: [
    (Story) => (
      <div style={{ width: "300px" }}>
        <div>Above</div>
        <Story />
        <div>Below</div>
      </div>
    ),
  ],
};

export const Vertical: Story = {
  args: {
    orientation: "vertical",
  },
  decorators: [
    (Story) => (
      <div style={{ height: "100px", display: "flex", alignItems: "center" }}>
        <div>Left</div>
        <Story />
        <div>Right</div>
      </div>
    ),
  ],
};

export const CustomStyle: Story = {
  args: {
    className: "bg-red-500",
  },
  decorators: [
    (Story) => (
      <div style={{ width: "300px" }}>
        <div>Above</div>
        <Story />
        <div>Below</div>
      </div>
    ),
  ],
};

export const WithContent: Story = {
  render: (args) => (
    <div className="space-y-1">
      <h4 className="text-sm font-medium leading-none">Radix Primitives</h4>
      <p className="text-sm text-muted-foreground">
        An open-source UI component library.
      </p>
      <Separator {...args} />
      <div className="flex h-5 items-center space-x-4 text-sm">
        <div>Blog</div>
        <Separator orientation="vertical" />
        <div>Docs</div>
        <Separator orientation="vertical" />
        <div>Source</div>
      </div>
    </div>
  ),
};
