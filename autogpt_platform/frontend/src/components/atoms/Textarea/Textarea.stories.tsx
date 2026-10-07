import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { Textarea } from "./Textarea";

const meta: Meta<typeof Textarea> = {
  title: "Atoms/Textarea",
  component: Textarea,
  tags: ["autodocs"],
  parameters: {
    a11y: { test: "error" },
    layout: "centered",
    docs: {
      description: {
        component:
          'A multi-line text field with the Input atom\'s field styles: label, hint, error, `rows`, and a length counter when `maxLength` is set. Works controlled (`value`) or uncontrolled (`defaultValue`). Replaces `Input type="textarea"`.',
      },
    },
  },
  decorators: [
    (Story) => (
      <div className="w-96">
        <Story />
      </div>
    ),
  ],
  argTypes: {
    size: { control: "inline-radio", options: ["sm", "md"] },
    rows: { control: { type: "number", min: 1 } },
    maxLength: { control: { type: "number", min: 1 } },
    hideLabel: { control: "boolean" },
    disabled: { control: "boolean" },
    error: { control: "text" },
    hint: { control: "text" },
  },
  args: {
    label: "Instructions",
    placeholder: "Tell the agent what to do",
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const WithHint: Story = {
  args: { hint: "Optional" },
};

export const WithCounter: Story = {
  args: { maxLength: 280, defaultValue: "Summarise the week's signups." },
};

export const WithError: Story = {
  args: {
    defaultValue: "Do it",
    error: "Add at least one full sentence.",
  },
};

export const Small: Story = {
  args: { size: "sm", rows: 2 },
};

export const Disabled: Story = {
  args: { disabled: true, defaultValue: "This field is locked." },
};

export const HiddenLabel: Story = {
  args: { hideLabel: true, placeholder: "Leave a comment" },
};

export const Controlled: Story = {
  render: function ControlledStory(args) {
    const [value, setValue] = useState("");
    return (
      <Textarea
        {...args}
        maxLength={140}
        value={value}
        onChange={(event) => setValue(event.target.value)}
      />
    );
  },
};
