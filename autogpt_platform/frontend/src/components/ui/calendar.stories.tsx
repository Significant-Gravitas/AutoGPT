import type { Meta, StoryObj } from "@storybook/react";

import { Calendar } from "./calendar";

const meta = {
  title: "UI/Calendar",
  component: Calendar,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    mode: {
      control: "select",
      options: ["single", "multiple", "range"],
    },
    selected: {
      control: "date",
    },
    showOutsideDays: {
      control: "boolean",
    },
  },
} satisfies Meta<typeof Calendar>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {},
};

export const SingleSelection: Story = {
  args: {
    mode: "single",
    selected: new Date(),
  },
};

export const MultipleSelection: Story = {
  args: {
    mode: "multiple",
    selected: [
      new Date(),
      new Date(new Date().setDate(new Date().getDate() + 5)),
    ],
  },
};

export const RangeSelection: Story = {
  args: {
    mode: "range",
    selected: {
      from: new Date(),
      to: new Date(new Date().setDate(new Date().getDate() + 7)),
    },
  },
};

export const HideOutsideDays: Story = {
  args: {
    showOutsideDays: false,
  },
};

export const CustomClassName: Story = {
  args: {
    className: "border rounded-lg shadow-lg",
  },
};
