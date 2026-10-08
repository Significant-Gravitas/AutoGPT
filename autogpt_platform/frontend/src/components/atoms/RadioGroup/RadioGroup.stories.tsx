import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { RadioGroup } from "./RadioGroup";

const OPTIONS = [
  {
    value: "compact",
    label: "Compact",
    description:
      "Friendly replies in plain prose. The machinery stays out of sight.",
  },
  {
    value: "technical",
    label: "Technical",
    description: "Every tool call, step and result, in full detail.",
  },
];

const meta = {
  title: "Atoms/RadioGroup",
  component: RadioGroup,
  tags: ["autodocs"],
  parameters: { layout: "padded", a11y: { test: "error" } },
  args: {
    label: "How do you want your agent to communicate?",
    options: OPTIONS,
    value: "compact",
    onValueChange: () => {},
  },
} satisfies Meta<typeof RadioGroup>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: function Render(args) {
    const [value, setValue] = useState(args.value);
    return <RadioGroup {...args} value={value} onValueChange={setValue} />;
  },
};

export const WithoutDescriptions: Story = {
  args: {
    options: OPTIONS.map(({ value, label }) => ({ value, label })),
  },
};

export const Disabled: Story = { args: { disabled: true } };

export const HiddenLabel: Story = { args: { hideLabel: true } };
