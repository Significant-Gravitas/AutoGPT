import { Text } from "@/components/atoms/Text/Text";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { fn } from "storybook/test";
import { TimeInput } from "./TimeInput";

const meta = {
  title: "Atoms/TimeInput",
  component: TimeInput,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="w-64">
        <Story />
      </div>
    ),
  ],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          'A native `type="time"` field styled to match the other inputs. `value` and `onChange` use an `"HH:MM"` string. Supports a label with hint (or a hidden label used as the aria-label), an error message, disabled, and `small`/`medium` sizes. The error slot always reserves its height so the layout does not jump.',
      },
    },
  },
  argTypes: {
    value: { control: "text", description: 'Time as "HH:MM"' },
    label: { control: "text", description: "Label text" },
    hint: { control: "text", description: "Hint shown next to the label" },
    hideLabel: {
      control: "boolean",
      description: "Hide the visible label and use it as the aria-label",
    },
    error: { control: "text", description: "Error message under the input" },
    disabled: { control: "boolean", description: "Disable the input" },
    size: {
      control: "select",
      options: ["sm", "md", "lg"],
      description: "Input height",
    },
  },
  args: {
    id: "daily-run-time",
    label: "Daily run time",
    onChange: fn(),
    size: "lg",
    disabled: false,
    hideLabel: false,
  },
} satisfies Meta<typeof TimeInput>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const WithValue: Story = {
  args: { value: "09:30" },
};

export const WithHint: Story = {
  args: { value: "09:30", hint: "24-hour clock" },
};

export const HiddenLabel: Story = {
  args: { hideLabel: true, value: "09:30" },
};

export const WithError: Story = {
  args: { value: "02:00", error: "Pick a time outside quiet hours" },
};

export const Disabled: Story = {
  args: { disabled: true, value: "09:30" },
};

export const Small: Story = {
  args: { size: "md", value: "09:30" },
};

export const Interactive: Story = {
  render: renderInteractive,
};

export const AllSizes: Story = {
  render: renderAllSizes,
};

function InteractiveTimeInput() {
  const [value, setValue] = useState("09:30");

  return (
    <div className="flex flex-col gap-2">
      <TimeInput
        id="interactive-time"
        label="Daily run time"
        value={value}
        onChange={setValue}
      />
      <Text variant="small" tone="muted">
        Value: {value || "none"}
      </Text>
    </div>
  );
}

function renderInteractive() {
  return <InteractiveTimeInput />;
}

function renderAllSizes() {
  return (
    <div className="flex flex-col gap-2">
      <TimeInput id="size-medium" label="Medium" size="lg" value="09:30" />
      <TimeInput id="size-small" label="Small" size="md" value="09:30" />
    </div>
  );
}
